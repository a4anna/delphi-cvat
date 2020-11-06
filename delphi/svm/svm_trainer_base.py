import io
import multiprocessing as mp
import numpy as np
import pickle
import queue
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any, List, Dict, Tuple, Union

import torch
from logzero import logger
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC, SVC
from sklearn.utils import parallel_backend

from delphi.context.model_trainer_context import ModelTrainerContext
from delphi.model import Model
from delphi.model_trainer import ModelTrainerBase
from delphi.svm.feature_cache import FeatureCache
from delphi.svm.feature_provider import FeatureProvider, BATCH_SIZE, get_worker_feature_provider, \
    set_worker_feature_provider
from delphi.svm.svc_wrapper import SVCWrapper
from delphi.svm.svm_model import SVMModel
from delphi.utils import log_exceptions, to_iter, bounded_iter

C_VALUES = [0.01, 0.1, 1, 10]
GAMMA_VALUES = [0.001, 0.01, 0.1, 1, 10]
F1_SCORER = make_scorer(f1_score, pos_label='1')


# return label, whether to preprocess, vector or (image, key)
@log_exceptions
def load_from_path(image_path: Path) -> Tuple[str, bool, Union[List[float], Any]]:
    split = image_path.parts
    label = split[-2]
    name = split[-1]

    key = get_worker_feature_provider().get_result_key_name(name)

    cached_vector = get_worker_feature_provider().get_cached_vector(key)
    if cached_vector is not None:
        return label, False, cached_vector
    else:
        with image_path.open('rb') as f:
            content = f.read()

        return label, True, (get_worker_feature_provider().preprocess(content).numpy(), key)


class SVMTrainerBase(ModelTrainerBase):

    def __init__(self, context: ModelTrainerContext, feature_extractor: str, cache: FeatureCache,
                 probability: bool):
        super().__init__()

        self.context = context
        self.feature_provider = FeatureProvider(feature_extractor, cache)
        self.probability = probability

    @property
    def should_sync_model(self) -> bool:
        return True

    def load_from_file(self, model_version: int, file: bytes) -> Model:
        bytes = io.BytesIO()
        bytes.write(file)
        bytes.seek(0)
        return SVMModel(pickle.load(bytes), model_version, self.feature_provider, self.probability)

    def _get_balanced(self, labels, examples):
        unique_labels  = np.unique(labels)
        example_ids_per_label = [np.where(labels==l)[0] for l in unique_labels]
        example_count = [len(ids) for ids in example_ids_per_label]
        min_count = min(example_count)
        assert min_count > 0, "Number of training example is less than 1"
        keep_ids = []
        for example_ids in example_ids_per_label:
            keep_ids.extend(np.random.choice(example_ids, min_count, replace=False))

        logger.debug("Training: NUM Labels {}, Count per Label {}, Total training examples {}".format(
            len(unique_labels), min_count, len(keep_ids)
        ))
        return labels[keep_ids], examples[keep_ids]

    def get_best_model(self, train_dir: Path, param_grid: List[Dict[str, Any]]) -> \
            Tuple[Union[LinearSVC, SVC, CalibratedClassifierCV], Any, float]:
        features = self._get_example_features(train_dir)
        flattened_examples = []
        flattened_labels = []

        for label in features:
            for feature_vector in features[label]:
                flattened_examples.append(feature_vector)
                flattened_labels.append(label)

        flattened_labels, flattened_examples = np.array(flattened_labels), np.array(flattened_examples)
        flattened_labels, flattened_examples = self._get_balanced(flattened_labels,
                                                    flattened_examples)

        with parallel_backend('threading'):
            grid_search = GridSearchCV(
                SVCWrapper(probability=self.probability),
                param_grid, n_jobs=4, scoring=F1_SCORER, verbose=0)
            grid_search.fit(flattened_examples, flattened_labels)

        logger.info('Best parameters found by grid search: {}'.format(grid_search.best_params_))
        return grid_search.best_estimator_.model, grid_search.best_params_, grid_search.best_score_

    # return label, vector or image, whether to preprocess
    def _get_example_features(self, example_dir: Path) -> Dict[str, List[List[float]]]:
        semaphore = threading.Semaphore(256)  # Make sure that the load function doesn't overload the consumer

        with mp.get_context('spawn').Pool(min(4, mp.cpu_count()), initializer=set_worker_feature_provider,
                                          initargs=(self.feature_provider.feature_extractor,
                                                    self.feature_provider.cache)) as pool:
            images = pool.imap_unordered(load_from_path, bounded_iter(example_dir.glob('*/*'), semaphore))
            feature_queue = queue.Queue()

            @log_exceptions
            def process_uncached():
                cached = 0
                uncached = 0
                batch = []
                for label, should_process, payload in images:
                    semaphore.release()
                    if should_process:
                        image, key = payload
                        batch.append((label,
                                      torch.from_numpy(image).to(self.feature_provider.device, non_blocking=True),
                                      key))
                        if len(batch) == BATCH_SIZE:
                            self._process_batch(batch, feature_queue)
                            batch = []
                        uncached += 1
                    else:
                        feature_queue.put((label, payload))
                        cached += 1

                if len(batch) > 0:
                    self._process_batch(batch, feature_queue)

                logger.info('{} cached examples, {} new examples preprocessed'.format(cached, uncached))
                feature_queue.put(None)

            threading.Thread(target=process_uncached, name='process-uncached-trainer').start()

            i = 0
            features = defaultdict(list)
            for feature in to_iter(feature_queue):
                i += 1
                features[feature[0]].append(feature[1])

            logger.info('Retrieved {} feature vectors'.format(i))

            return features

    # label, image, key -> label, vector
    def _process_batch(self, items: List[Tuple[str, torch.Tensor, str]], feature_queue: queue.Queue) -> None:
        keys = [i[2] for i in items]
        tensor = torch.stack([i[1] for i in items])
        results = self.feature_provider.cache_and_get(keys, tensor, True)
        for item in items:
            feature_queue.put((item[0], results[item[2]]))
