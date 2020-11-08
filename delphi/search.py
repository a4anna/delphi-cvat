import io
import multiprocessing as mp
import os
import queue
import threading
import time
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Iterable, Tuple

import numpy as np
import tensorboard
from google.protobuf.any_pb2 import Any
from logzero import logger
from sklearn.metrics import classification_report, average_precision_score
from sklearn.metrics import precision_recall_curve, confusion_matrix
from torch.utils.tensorboard import SummaryWriter

from delphi.condition.model_condition import ModelCondition
from delphi.context.data_manager_context import DataManagerContext
from delphi.context.model_trainer_context import ModelTrainerContext
from delphi.data_manager import DataManager
from delphi.delphi_stub import DelphiStub
from delphi.model import Model
from delphi.model_trainer import ModelTrainer, TrainingStyle
from delphi.object_provider import ObjectProvider
from delphi.proto.internal_pb2 import ExampleMetadata, TestResult, StageModelRequest, TrainModelRequest, \
    DiscardModelRequest, PromoteModelRequest, SubmitTestRequest, ValidateTestResultsRequest, SubmitTestVersion
from delphi.proto.delphi_pb2 import ModelStats, ModelMetrics, ModelArchive, SearchId, LabeledExample, \
    ExampleSet
from delphi.result_provider import ResultProvider
from delphi.retrain.retrain_policy import RetrainPolicy
from delphi.retrieval.retriever import Retriever
from delphi.selection.selector import Selector
from delphi.utils import log_exceptions, to_iter


class Search(DataManagerContext, ModelTrainerContext):
    trainers: List[ModelCondition]

    def __init__(self, id: SearchId, node_index: int, nodes: List[DelphiStub], retrain_policy: RetrainPolicy,
                 only_use_better_models: bool, root_dir: Path, port: int, retriever: Retriever, selector: Selector,
                 has_initial_examples: bool):
        self._id = id
        self._node_index = node_index
        self._nodes = nodes

        self._retrain_policy = retrain_policy
        self._data_dir = root_dir
        self._tb_dir = root_dir / 'tb'

        self._only_use_better_models = only_use_better_models
        self._port = port

        self.retriever = retriever
        self.selector = selector

        # Indicates that the search will seed the strategy with an initial set of examples. The strategy should
        # therefore hold off on returning inference results until its underlying model is trained
        self._has_initial_examples = has_initial_examples

        self._model: Optional[Model] = None
        self._tb_writer: Optional[SummaryWriter] = None
        self._data_manager = DataManager(self)

        self._model_stats = ModelStats(version=-1)
        self._model_lock = threading.Lock()
        self._initial_model_event = threading.Event()
        self._model_event = threading.Event()
        self._last_trained_version = -1

        self._test_results: Dict[int, List[List[Tuple[int, float]]]] = defaultdict(list)
        self._results_condition = mp.Condition()

        self._staged_models: Dict[int, Model] = {}
        self._staged_model_condition = mp.Condition()

        self._abort_event = threading.Event()
        self._pause_flag = False

        if self._node_index == 0:
            threading.Thread(target=self._train_thread, name='train-model').start()

    def pause(self):
        self._pause_flag = True

    def _is_paused(self):
        return self._pause_flag

    def restart(self):
        self._pause_flag = False

    def infer(self, requests: Iterable[ObjectProvider]) -> Iterable[ResultProvider]:
        while self._is_paused(): # Wait till restart
            pass

        with self._model_lock:
            model = self._model

        if model is None:
            if self._has_initial_examples:
                # Wait for the model to train before inferencing
                self._initial_model_event.wait()
            else:
                # Pass all examples until user has labeled enough examples to train a model
                for request in requests:
                    yield ResultProvider(request.id, '1', 0, None, request.attributes, request.gt)
                return

        with self._model_lock:
            model = self._model

        yield from model.infer(requests)

    def add_labeled_examples(self, examples: Iterable[LabeledExample]) -> None:
        self._data_manager.add_labeled_examples(examples)

    def get_examples(self, example_set: ExampleSet, node_index: int) -> Iterable[ExampleMetadata]:
        return self._data_manager.get_example_stream(example_set, node_index)

    def get_example(self, example_set: ExampleSet, label: str, example: str) -> Path:
        return self._data_manager.get_example_path(example_set, label, example)

    def get_model_stats(self) -> ModelStats:
        # We assume that the master is the source of truth
        if self._node_index != 0:
            return self._nodes[0].api.GetModelStats(self._id)

        with self._model_lock:
            return self._model_stats if self._model_stats is not None else ModelStats(version=self._model.version)

    def import_model(self, model_version: int, file: bytes) -> None:
        assert self._node_index == 0
        assert len(self.trainers) == 1

        model = self.trainers[0].trainer.load_from_file(model_version, file)
        self._score_and_set_model(model, True)

    def export_model(self) -> ModelArchive:
        assert self._node_index == 0

        memory_file = io.BytesIO()
        with self._model_lock:
            assert self._model is not None
            model = self._model
            model_version = self._model.version

        with zipfile.ZipFile(memory_file, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr('model', model.get_bytes())
            if self._tb_writer is not None:
                for file in self._tb_dir.iterdir():
                    zf.write(file, arcname=os.path.join('tensorboard', file.name))

        memory_file.seek(0)
        return ModelArchive(version=model_version, content=memory_file.getvalue())

    def train_model(self, trainer_index: int) -> None:
        assert self._node_index != 0 \
               and self.trainers[trainer_index].trainer.training_style is TrainingStyle.DISTRIBUTED
        threading.Thread(target=self._train_model_slave_thread, args=(trainer_index,), name='train-model').start()

    def stage_model(self, model_version: int, trainer_index: int, file: bytes) -> None:
        assert self._node_index != 0
        model = self.trainers[trainer_index].trainer.load_from_file(model_version, file)
        with self._staged_model_condition:
            self._staged_models[model_version] = model
            self._staged_model_condition.notify_all()

    def validate_test_results(self, model_version: int) -> None:
        assert self._node_index != 0
        threading.Thread(target=self._validate_test_results_thread, args=(model_version,),
                         name='validate-test_results').start()

    def submit_test_results(self, results: Iterable[TestResult], model_version: int) -> None:
        assert self._node_index == 0
        model_results = [(int(result.label), result.score) for result in results]

        with self._results_condition:
            self._test_results[model_version].append(model_results)
            remaining = len(self._nodes) - len(self._test_results[model_version])
            logger.info('Received {} validation results ({} nodes remaining)'.format(len(model_results), remaining))
            if remaining == 0:
                self._results_condition.notify_all()

    def promote_model(self, model_version: int):
        model = self._get_staging_model(model_version)
        with self._staged_model_condition:
            del self._staged_models[model_version]

        with self._model_lock:
            should_notify = self._model is None
            self._model = model
            self._last_trained_version = model_version

        self.selector.new_model(model)

        if should_notify:
            self._initial_model_event.set()

        logger.info('Promoted model version {}'.format(model_version))

    def discard_model(self, model_version: int):
        with self._staged_model_condition:
            del self._staged_models[model_version]

        with self._model_lock:
            self._last_trained_version = model_version

        logger.info('Discarded model version {}'.format(model_version))

    def message_internal(self, trainer_index: int, request: Any) -> Any:
        return self.trainers[trainer_index].trainer.message_internal(request)

    def reset(self, train_only: bool) -> None:
        self._data_manager.reset(train_only)

        with self._model_lock:
            self._model = None
            self._model_stats = ModelStats(version=-1)
            self._last_trained_version = -1

        self.selector.new_model(None)

    def get_last_trained_version(self) -> int:
        with self._model_lock:
            return self._last_trained_version

    @property
    def node_index(self) -> int:
        return self._node_index

    @property
    def nodes(self) -> List[DelphiStub]:
        return self._nodes

    @property
    def port(self) -> int:
        return self._port

    @property
    def tb_writer(self) -> SummaryWriter:
        if self._tb_writer is None:
            self._start_tensorboard()
            self._tb_writer = SummaryWriter(str(self._tb_dir))

        return self._tb_writer

    @property
    def search_id(self) -> SearchId:
        return self._id

    @property
    def data_dir(self) -> Path:
        return self._data_dir

    def get_active_trainers(self) -> List[ModelTrainer]:
        return [x.trainer for x in self.trainers]

    def new_examples_callback(self, new_positives: int, new_negatives: int) -> None:
        self._retrain_policy.update(new_positives, new_negatives)
        if self._retrain_policy.should_retrain():
            should_retrain = True
        else:
            with self._model_lock:
                model = self._model
            should_retrain = model is None

        if should_retrain:
            self._retrain_policy.reset()
            self._model_event.set()

    def start(self) -> None:
        try:
            self.retriever.start()
            threading.Thread(target=self._retriever_thread, name='get-objects').start()
        except Exception as e:
            self.retriever.stop()
            self.selector.finish()
            raise e

    def stop(self) -> None:
        logger.info("Stop called")
        self._abort_event.set()
        for trainer in self.trainers:
            trainer.close()

    def _objects_for_model_version(self) -> Iterable[Optional[ObjectProvider]]:
        with self._model_lock:
            starting_version = self._model.version if self._model is not None else None

        logger.info('Starting evaluation with model version {}'.format(starting_version))

        for retriever_object in self.retriever.get_objects():

            if self._is_paused(): # Wait till restart
                return

            with self._model_lock:
                version = self._model.version if self._model is not None else None
            if version != starting_version:
                logger.info('Done evaluating with model version {} (new version {} available)'.format(starting_version,
                                                                                                      version))
                return
            yield retriever_object

        self._abort_event.set()

    @log_exceptions
    def _retriever_thread(self) -> None:
        try:
            if self._has_initial_examples:
                self._initial_model_event.wait()

            while True:
                for result in self.infer(self._objects_for_model_version()):
                    if result is None:
                        break
                    if self._abort_event.is_set():
                        return
                    self.selector.add_result(result)
        finally:
            self.retriever.stop()
            self.selector.finish()

    @log_exceptions
    def _train_thread(self) -> None:
        while True:
            try:
                self._model_event.wait()
                time.sleep(5)  # Wait a bit to see if examples come from other nodes and avoid retraining multiple times
                self._model_event.clear()
                self._train_model()

            except Exception as e:
                logger.exception(e)

    def _train_model(self) -> None:
        assert self._node_index == 0
        train_start = time.time()
        self._pause_flag = True

        with self._data_manager.get_examples(ExampleSet.LABELED) as train_dir:
            example_counts = {}
            for label in train_dir.iterdir():
                example_counts[label] = len(list(label.iterdir()))

            logger.info("Number of training examples {}".format(example_counts))

            trainer_index = None

            with self._model_lock:
                model_stats = self._model_stats

            for i in range(len(self.trainers)):
                if self.trainers[i].is_satisfied(example_counts, model_stats):
                    trainer_index = i
                    break

            if trainer_index is None:
                logger.info('Current conditions do not match any trainer - aborting')
                self._pause_flag = False
                return

            if self.trainers[trainer_index].trainer.training_style is TrainingStyle.DISTRIBUTED \
                    and len(self._nodes) > 0:
                for node in self._nodes[1:]:
                    node.internal.TrainModel(TrainModelRequest(searchId=self._id, trainerIndex=trainer_index))

            model = self.trainers[trainer_index].trainer.train_model(train_dir)

        eval_start = time.time()
        logger.info('Trained model in {:.3f} seconds'.format(eval_start - train_start))
        self._pause_flag = False
        self._score_and_set_model(model, self.trainers[trainer_index].trainer.should_sync_model)
        logger.info('Evaluated model in {:.3f} seconds'.format(time.time() - eval_start))

    def _score_and_set_model(self, model: Model, should_stage: bool) -> None:
        with self._data_manager.get_examples(ExampleSet.TEST) as test_dir:
            if len(self._nodes) > 0:
                for node in self._nodes[1:]:
                    if should_stage:
                        node.internal.StageModel(
                            StageModelRequest(searchId=self._id, version=model.version, content=model.get_bytes()))
                        logger.info('Staged model on node {}'.format(node.url))

                    node.internal.ValidateTestResults(
                        ValidateTestResultsRequest(searchId=self._id, version=model.version))
                    logger.info('Started validation for model version {} on node {}'.format(model.version, node.url))

            results = []

            def callback_fn(target: int, pred: float):
                results.append((target, pred))

            model.infer_dir(test_dir, callback_fn)

            with self._results_condition:
                test_results = self._test_results[model.version]
                test_results.append(results)
                while len(test_results) < len(self._nodes):
                    self._results_condition.wait()

                del self._test_results[model.version]
        model_stats = None
        targets = []
        preds = []
        labels = set()
        for node_results in test_results:
            for result in node_results:
                targets.append(result[0])
                preds.append(result[1])
                labels.add(results[0])

        # Only create model stats if we have sufficient test set data
        if len(labels) > 0:
            model_stats = self.create_model_stats(model.version, targets, preds, model.scores_are_probabilities)

        with self._model_lock:
            should_notify = self._model is None
            if self._only_use_better_models \
                    and model_stats is not None \
                    and self._model_stats is not None \
                    and self._model_stats.auc > model_stats.auc:
                better_old_score = model_stats.auc
            else:
                self._model = model
                self._model_stats = model_stats
                better_old_score = None

            self._last_trained_version = model.version

        if better_old_score is not None:
            logger.info('New model has worse test AUC score than previous one - discarding (new: {}, old: {})'
                        .format(model_stats.auc, better_old_score))
            if len(self._nodes) > 0:
                for node in self._nodes[1:]:
                    node.internal.DiscardModel(DiscardModelRequest(searchId=self._id, version=model.version))
                    logger.info('Discarded model version {} on node {}'.format(model.version, node.url))
        else:
            self.selector.new_model(model)

            if should_notify:
                self._initial_model_event.set()

            if len(self._nodes) > 0:
                for node in self._nodes[1:]:
                    node.internal.PromoteModel(PromoteModelRequest(searchId=self._id, version=model.version))
                    logger.info('Promoted model on node {} to version {}'.format(node.url, model.version))

    @log_exceptions
    def _train_model_slave_thread(self, trainer_index: int) -> None:
        logger.info('Executing train request')

        with self._data_manager.get_examples(ExampleSet.LABELED) as train_dir:
            train_start = time.time()
            model = self.trainers[trainer_index].trainer.train_model(train_dir)
            logger.info('Trained model in {:.3f} seconds'.format(time.time() - train_start))

        if not self.trainers[trainer_index].trainer.should_sync_model:
            with self._staged_model_condition:
                self._staged_models[model.version] = model
                self._staged_model_condition.notify_all()

    @log_exceptions
    def _validate_test_results_thread(self, model_version: int) -> None:
        logger.info('Executing validation request')

        model = self._get_staging_model(model_version)
        result_queue = queue.Queue()

        future = self.nodes[0].internal.SubmitTestResults.future(to_iter(result_queue))
        result_queue.put(SubmitTestRequest(version=SubmitTestVersion(searchId=self._id, version=model_version)))

        def store_result(target: int, pred: float):
            result_queue.put(SubmitTestRequest(result=TestResult(label=str(target), score=pred)))

        eval_start = time.time()
        with self._data_manager.get_examples(ExampleSet.TEST) as test_dir:
            model.infer_dir(test_dir, store_result)

        result_queue.put(None)
        future.result()

        logger.info('Evaluated model in {:.3f} seconds'.format(time.time() - eval_start))
        logger.info('Submitted test results for model version {}'.format(model.version))

    def _get_staging_model(self, model_version: int) -> Model:
        while True:
            with self._staged_model_condition:
                if model_version in self._staged_models:
                    model = self._staged_models[model_version]
                    break
                else:
                    logger.info('Waiting for model version {}'.format(model_version))
                self._staged_model_condition.wait()
        return model

    def _start_tensorboard(self) -> None:
        tb = tensorboard.program.TensorBoard()
        tb_port = self._port + 1
        for i in range(10):
            tb.configure(argv=[None, '--logdir', self._tb_dir, '--host', '0.0.0.0', '--port', str(tb_port)])
            logger.info('Trying to launch Tensorboard on port {}'.format(tb_port))
            try:
                tb.launch()
                logger.info('Started Tensorboard on port {}'.format(tb_port))
                return
            except tensorboard.program.TensorBoardPortInUseError:
                tb_port += 1
                logger.warn('Failed to start Tensorboard on port {} (port already in use)'.format(tb_port))

        logger.error('Failed to start Tensorboard')

    @staticmethod
    def create_model_stats(version: int, target: List[int], pred: List[float], is_probability: bool) \
            -> Optional[ModelStats]:
        assert len(target) == len(pred)
        pred = np.array(pred)
        target = np.array(target)

        ap = average_precision_score(target, pred, average=None)
        precision, recall, thresholds = precision_recall_curve(target, pred)
        f1_score = np.nan_to_num(2 * (precision * recall) / (precision + recall))
        f1_best_idx = np.argmax(f1_score)

        best_threshold = thresholds[f1_best_idx]
        pred_best_f1 = np.where(pred >= best_threshold, 1, 0)
        logger.info('Test AUC: {}'.format(ap))

        logger.info(
            'Test classification report (ideal threshold):\n{}'.format(classification_report(target, pred_best_f1)))
        stats_best_f1 = classification_report(target, pred_best_f1, output_dict=True)

        # Only predictions for a single class - can't build a confusion matrix
        if '0' not in stats_best_f1 or '1' not in stats_best_f1:
            return None

        _, fp_best_f1, fn_best_f1, tp_best_f1 = confusion_matrix(target, pred_best_f1).ravel()

        pred = np.where(pred >= 0.5, 1, 0) if is_probability else np.where(pred > 0, 1, 0)
        logger.info(
            'Test classification report (0.5 threshold):\n{}'.format(classification_report(target, pred)))
        stats = classification_report(target, pred, output_dict=True)

        if '0' not in stats or '1' not in stats:
            return None

        _, fp, fn, tp = confusion_matrix(target, pred).ravel()

        return ModelStats(
            testExamples=len(pred),
            auc=ap,
            validationMetrics=ModelMetrics(
                truePositives=tp.item(),
                falsePositives=fp.item(),
                falseNegatives=fn.item(),
                precision=stats['1']['precision'],
                recall=stats['1']['recall'],
                f1Score=stats['1']['f1-score']
            ),
            idealMetrics=ModelMetrics(
                truePositives=tp_best_f1.item(),
                falsePositives=fp_best_f1.item(),
                falseNegatives=fn_best_f1.item(),
                precision=stats_best_f1['1']['precision'],
                recall=stats_best_f1['1']['recall'],
                f1Score=stats_best_f1['1']['f1-score']
            ),
            bestThreshold=best_threshold.item(),
            precisions=[item.item() for item in precision],
            recalls=[item.item() for item in recall],
            thresholds=[item.item() for item in thresholds],
            version=version
        )
