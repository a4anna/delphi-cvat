import multiprocessing as mp
import pickle
import queue
import threading
from pathlib import Path
from typing import Callable, Iterable, List, Tuple, Any, Union

import torch
from logzero import logger
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.svm import LinearSVC, SVC

from delphi.model import Model
from delphi.object_provider import ObjectProvider
from delphi.result_provider import ResultProvider
from delphi.attribute_provider import SimpleAttributeProvider
from delphi.svm.feature_provider import FeatureProvider, BATCH_SIZE, get_worker_feature_provider, \
    set_worker_feature_provider
from delphi.utils import log_exceptions, bounded_iter

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


# return object_provider, whether to preprocess, vector or (image, key)
@log_exceptions
def load_from_path(image_path: Path) -> Tuple[ObjectProvider, bool, Union[List[float], Any]]:
    split = image_path.parts
    label = split[-2]
    name = split[-1]
    object_id = '{}/{}'.format(label, name)

    key = get_worker_feature_provider().get_result_key_name(name)
    cached_vector = get_worker_feature_provider().get_cached_vector(key)
    provider = ObjectProvider(object_id, b'', SimpleAttributeProvider({}), False)
    if cached_vector is not None:
        return provider, False, cached_vector
    else:
        with image_path.open('rb') as f:
            content = f.read()

        return provider, True, (get_worker_feature_provider().preprocess(content).numpy(), key)


# return object_provider, whether to preprocess, vector or (image, key)
@log_exceptions
def load_from_content(request: ObjectProvider) -> Tuple[ObjectProvider, bool, Union[List[float], Any]]:
    content = request.content
    key = get_worker_feature_provider().get_result_key_content(content)
    cached_vector = get_worker_feature_provider().get_cached_vector(key)
    if cached_vector is not None:
        return request, False, cached_vector
    else:
        return request, True, (get_worker_feature_provider().preprocess(content).numpy(), key)


class SVMModel(Model):

    def __init__(self, svc: Union[LinearSVC, SVC, CalibratedClassifierCV, VotingClassifier], version: int,
                 feature_provider: FeatureProvider, probability: bool):
        self._svc = svc
        self._version = version
        self._feature_provider = feature_provider
        self._probability = probability
        self._system_examples = []
        self._system_examples_lock = threading.Lock()

    @property
    def version(self) -> int:
        return self._version

    def infer(self, requests: Iterable[ObjectProvider]) -> Iterable[ResultProvider]:
        semaphore = threading.Semaphore(256)  # Make sure that the load function doesn't overload the consumer

        with mp.get_context('spawn').Pool(min(4, mp.cpu_count()), initializer=set_worker_feature_provider,
                                          initargs=(self._feature_provider.feature_extractor,
                                                    self._feature_provider.cache)) as pool:
            images = pool.imap_unordered(load_from_content, bounded_iter(requests, semaphore))
            yield from self._infer_inner(images, semaphore)

    def infer_dir(self, directory: Path, callback_fn: Callable[[int, float], None]) -> None:
        semaphore = threading.Semaphore(256)  # Make sure that the load function doesn't overload the consumer

        with mp.get_context('spawn').Pool(min(4, mp.cpu_count()), initializer=set_worker_feature_provider,
                                          initargs=(self._feature_provider.feature_extractor,
                                                    self._feature_provider.cache)) as pool:
            images = pool.imap_unordered(load_from_path, bounded_iter(directory.glob('*/*'), semaphore))
            results = self._infer_inner(images, semaphore)

            i = 0
            for result in results:
                i += 1
                # TODO(hturki): Should we get the target label in a less hacky way?
                callback_fn(int(result.id.split('/')[-2]), result.score)
                if i % 1000 == 0:
                    logger.info('{} examples scored so far'.format(i))

    def get_bytes(self) -> bytes:
        return pickle.dumps(self._svc)

    @property
    def scores_are_probabilities(self) -> bool:
        return self._probability

    def _infer_inner(self, images: Iterable[Tuple[ObjectProvider, bool, Union[List[float], Any]]],
                     semaphore: threading.Semaphore) -> Iterable[ResultProvider]:
        feature_queue = queue.Queue(1000)

        @log_exceptions
        def process_uncached():
            cached = 0
            uncached = 0
            batch = []
            for provider, should_process, payload in images:
                semaphore.release()
                if should_process:
                    image, key = payload
                    batch.append((provider,
                                  torch.from_numpy(image).to(self._feature_provider.device, non_blocking=True),
                                  key))
                    if len(batch) == BATCH_SIZE:
                        self._process_batch(batch, feature_queue)
                        batch = []
                    uncached += 1
                else:
                    feature_queue.put((provider, payload))
                    cached += 1

            if len(batch) > 0:
                self._process_batch(batch, feature_queue)

            logger.info('{} cached examples, {} new examples preprocessed'.format(cached, uncached))
            feature_queue.put(None)

        threading.Thread(target=process_uncached, name='process-uncached-model').start()

        scored = 0
        queue_finished = False
        while not queue_finished:
            providers = []
            features = []

            item = feature_queue.get()
            if item is None:
                break

            providers.append(item[0])
            features.append(item[1])

            while True:
                try:
                    item = feature_queue.get(block=False)
                    if item is None:
                        queue_finished = True
                        break

                    providers.append(item[0])
                    features.append(item[1])
                except queue.Empty:
                    break

            if len(features) == 0:
                continue

            scores = self._svc.predict_proba(features) if self._probability else self._svc.decision_function(
                features)
            scored += len(providers)
            for i in range(len(providers)):
                if self._probability:
                    score = scores[i][1]
                    label = '1' if score >= 0.5 else '0'
                else:
                    score = scores[i]
                    label = '1' if score > 0 else '0'

                yield ResultProvider(providers[i].id, label, score, self.version, providers[i].attributes,
                                     providers[i].gt)

        logger.info('{} examples scored'.format(scored))

    # provider, image, key -> provider, vector
    def _process_batch(self, items: List[Tuple[ObjectProvider, torch.Tensor, str]], feature_queue: queue.Queue) -> None:
        keys = [i[2] for i in items]
        tensor = torch.stack([i[1] for i in items])
        results = self._feature_provider.cache_and_get(keys, tensor, True)
        for item in items:
            feature_queue.put((item[0], results[item[2]]))
