import queue
import threading
from pathlib import Path
from typing import Iterable

import grpc
from google.protobuf import json_format
from google.protobuf.empty_pb2 import Empty
from google.protobuf.wrappers_pb2 import Int64Value
from logzero import logger

from delphi.attribute_provider import SimpleAttributeProvider
from delphi.condition.bandwidth_condition import BandwidthCondition
from delphi.condition.examples_per_label_condition import ExamplesPerLabelCondition
from delphi.condition.test_auc_condition import TestAucCondition
from delphi.context.model_trainer_context import ModelTrainerContext
from delphi.delphi_stub import DelphiStub
from delphi.object_provider import ObjectProvider
from delphi.proto.delphi_pb2 import InferRequest, InferResult, ModelStats, \
    ImportModelRequest, ModelArchive, LabeledExampleRequest, SearchId, \
    AddLabeledExampleIdsRequest, LabeledExample, DelphiObject, GetObjectsRequest, SearchStats, SearchInfo, \
    CreateSearchRequest
from delphi.proto.delphi_pb2 import RetrainPolicyConfig, SVMMode, SVMConfig, Dataset, \
    SelectorConfig, ReexaminationStrategyConfig
from delphi.proto.delphi_pb2_grpc import DelphiServiceServicer
from delphi.retrain.absolute_threshold_policy import AbsoluteThresholdPolicy
from delphi.retrain.percentage_threshold_policy import PercentageThresholdPolicy
from delphi.retrain.retrain_policy import RetrainPolicy
from delphi.retrieval.retriever import Retriever
from delphi.retrieval.directory_retriever import DirectoryRetriever
from delphi.search import Search
from delphi.search_manager import SearchManager
from delphi.selection.full_reexamination_strategy import FullReexaminationStrategy
from delphi.selection.no_reexamination_strategy import NoReexaminationStrategy
from delphi.selection.reexamination_strategy import ReexaminationStrategy
from delphi.selection.selector import Selector
from delphi.selection.threshold_selector import ThresholdSelector
from delphi.selection.top_reexamination_strategy import TopReexaminationStrategy
from delphi.selection.topk_selector import TopKSelector
from delphi.svm.distributed_svm_trainer import DistributedSVMTrainer
from delphi.svm.ensemble_svm_trainer import EnsembleSVMTrainer
from delphi.svm.feature_cache import FeatureCache
from delphi.svm.svm_trainer import SVMTrainer
from delphi.svm.svm_trainer_base import SVMTrainerBase
from delphi.utils import to_iter

ATTR_DATA = ''


class DelphiServicer(DelphiServiceServicer):

    def __init__(self, manager: SearchManager, root_dir: Path, feature_cache: FeatureCache, port: int):
        self._manager = manager
        self._root_dir = root_dir
        self._feature_cache = feature_cache
        self._port = port

    def CreateSearch(self, request: CreateSearchRequest, context: grpc.ServicerContext) -> Empty:
        try:
            logger.info("Create Search called")
            retrain_policy = self._get_retrain_policy(request.retrainPolicy)

            nodes = [DelphiStub(node) for node in request.nodes]
            search = Search(request.searchId, request.nodeIndex, nodes,
                            retrain_policy, request.onlyUseBetterModels,
                            self._root_dir,
                            self._port, self._get_retriever(request.dataset),
                            self._get_selector(request.selector), request.hasInitialExamples)

            trainers = []
            for i in range(len(request.trainStrategy)):
                if request.trainStrategy[i].HasField('examplesPerLabel'):
                    condition_builder = lambda x: ExamplesPerLabelCondition(
                        request.trainStrategy[i].examplesPerLabel.count,
                        x)
                    model = request.trainStrategy[i].examplesPerLabel.model
                elif request.trainStrategy[i].HasField('testAuc'):
                    condition_builder = lambda x: TestAucCondition(request.trainStrategy[i].testAuc.threshold, x)
                    model = request.trainStrategy[i].testAuc.model
                elif request.trainStrategy[i].HasField('bandwidth'):
                    bandwidth_config = request.trainStrategy[i].bandwidth
                    condition_builder = lambda x: BandwidthCondition(request.nodeIndex, nodes,
                                                                     bandwidth_config.thresholdMbps,
                                                                     bandwidth_config.refreshSeconds, x)
                    model = request.trainStrategy[i].bandwidth.model
                else:
                    raise NotImplementedError(
                        'unknown condition: {}'.format(json_format.MessageToJson(request.trainStrategy[i])))

                if model.HasField('svm'):
                    trainer = self._get_svm_trainer(search, request.searchId, i, model.svm)
                else:
                    raise NotImplementedError('unknown model: {}'.format(json_format.MessageToJson(model)))

                trainers.append(condition_builder(trainer))

            search.trainers = trainers
            self._manager.set_search(request.searchId, search, request.metadata)

            logger.info('Create search with id {} and parameters:\n{}'.format(request.searchId.value,
                                                                              json_format.MessageToJson(request)))
            return Empty()
        except Exception as e:
            logger.exception(e)
            raise e

    def StartSearch(self, request: SearchId, context: grpc.ServicerContext) -> Empty:
        try:
            self._manager.get_search(request).start()
            logger.info('Starting search with id {}'.format(request.value))
            return Empty()
        except Exception as e:
            logger.exception(e)
            raise e

    def PauseSearch(self, request: SearchId, context: grpc.ServicerContext) -> Empty:
        try:
            logger.info('Pausing search with id {}'.format(request.value))
            search = self._manager.get_search(request).pause()
            return Empty()
        except Exception as e:
            logger.exception(e)
            raise e

    def RestartSearch(self, request: SearchId, context: grpc.ServicerContext) -> Empty:
        try:
            logger.info('Restarting search with id {}'.format(request.value))
            search = self._manager.get_search(request).restart()
            return Empty()
        except Exception as e:
            logger.exception(e)
            raise e

    def StopSearch(self, request: SearchId, context: grpc.ServicerContext) -> Empty:
        try:
            logger.info('Stopping search with id {}'.format(request.value))
            search = self._manager.remove_search(request)
            search.stop()
            return Empty()
        except Exception as e:
            logger.exception(e)
            raise e

    def GetSearches(self, request: SearchId, context: grpc.ServicerContext) -> Iterable[SearchInfo]:
        try:
            for search_id, metadata in self._manager.get_searches():
                yield SearchInfo(searchId=search_id, metadata=metadata)
        except Exception as e:
            logger.exception(e)
            raise e

    #TODO Remove
    def GetMessages(self, request_iterator, context) -> Iterable[str]:
        for message in request_iterator:
            yield message

    def GetResults(self, request: SearchId, context: grpc.ServicerContext) -> Iterable[InferResult]:
        try:
            while True:
                result = self._manager.get_search(request).selector.get_result()
                if result is None:
                    return
                yield InferResult(objectId=result.id, label=result.label, score=result.score,
                                  modelVersion=result.model_version, attributes=result.attributes.get())
        except Exception as e:
            logger.exception(e)
            raise e

    def GetObjects(self, request: GetObjectsRequest, context: grpc.ServicerContext) -> Iterable[DelphiObject]:
        try:
            retriever = self._get_retriever(request.dataset)
            try:
                retriever.start()
                for object_id in request.objectIds:
                    yield retriever.get_object(object_id, request.attributes)
            finally:
                retriever.stop()
        except Exception as e:
            logger.exception(e)
            raise e

    def Infer(self, request: Iterable[InferRequest], context: grpc.ServicerContext) -> Iterable[InferResult]:
        try:
            search_id = next(request).searchId
            for result in self._manager.get_search(search_id).infer(
                    ObjectProvider(x.object.objectId, x.object.content, SimpleAttributeProvider(x.object.attributes),
                                   False)
                    for x in request):
                yield InferResult(objectId=result.id, label=result.label, score=result.score,
                                  modelVersion=result.model_version, attributes=result.attributes.get())
        except Exception as e:
            logger.exception(e)
            raise e

    def AddLabeledExamples(self, request: Iterable[LabeledExampleRequest], context: grpc.ServicerContext) -> Empty:
        try:
            search_id = next(request).searchId
            self._manager.get_search(search_id).add_labeled_examples(x.example for x in request)
            return Empty()
        except Exception as e:
            logger.exception(e)
            raise e

    def AddLabeledExampleIds(self, request: AddLabeledExampleIdsRequest, context: grpc.ServicerContext) -> Empty:
        try:
            search = self._manager.get_search(request.searchId)

            examples = queue.Queue()
            exceptions = []

            def get_examples():
                try:
                    for object_id in request.examples:
                        example = search.retriever.get_object(object_id, [ATTR_DATA])
                        examples.put(LabeledExample(label=request.examples[object_id], content=example.content))
                except Exception as e:
                    exceptions.append(e)
                finally:
                    examples.put(None)

            threading.Thread(target=get_examples, name='get-examples').start()

            search.add_labeled_examples(to_iter(examples))

            if len(exceptions) > 0:
                raise exceptions[0]

            return Empty()
        except Exception as e:
            logger.exception(e)
            raise e

    def GetSearchStats(self, request: SearchId, context: grpc.ServicerContext) -> SearchStats:
        try:
            search = self._manager.get_search(request)
            retriever_stats = search.retriever.get_stats()
            selector_stats = search.selector.get_stats()
            passed_objects = selector_stats.passed_objects
            return SearchStats(totalObjects=retriever_stats.total_objects,
                               processedObjects=retriever_stats.dropped_objects + selector_stats.processed_objects,
                               droppedObjects=retriever_stats.dropped_objects + selector_stats.dropped_objects,
                               passedObjects=Int64Value(value=passed_objects) if passed_objects is not None else None,
                               falseNegatives=retriever_stats.false_negatives + selector_stats.false_negatives)
        except Exception as e:
            logger.exception(e)
            raise e

    def GetModelStats(self, request: SearchId, context: grpc.ServicerContext) -> ModelStats:
        try:
            return self._manager.get_search(request).get_model_stats()
        except Exception as e:
            logger.exception(e)
            raise e

    def ImportModel(self, request: ImportModelRequest, context: grpc.ServicerContext) -> Empty:
        try:
            self._manager.get_search(request.searchId).import_model(request.version, request.content)
            return Empty()
        except Exception as e:
            logger.exception(e)
            raise e

    def ExportModel(self, request: Empty, context: grpc.ServicerContext) -> ModelArchive:
        try:
            return self._manager.get_search(request.searchId).export_model()
        except Exception as e:
            logger.exception(e)
            raise e

    def StopSearch(self, request: SearchId, context: grpc.ServicerContext) -> Empty:
        try:
            logger.info('Stopping search with id {}'.format(request.value))
            search = self._manager.remove_search(request)
            search.stop()
            return Empty()
        except Exception as e:
            logger.exception(e)
            raise e

    def stop(self) -> None:
        try:
            for search_id, _ in self._manager.get_searches():
                search = self._manager.remove_search(search_id)
                search.stop()
        except Exception as e:
            logger.exception(e)
            raise e

    def _get_retrain_policy(self, retrain_policy: RetrainPolicyConfig) -> RetrainPolicy:
        if retrain_policy.HasField('absolute'):
            return AbsoluteThresholdPolicy(retrain_policy.absolute.threshold, retrain_policy.absolute.onlyPositives)
        elif retrain_policy.HasField('percentage'):
            return PercentageThresholdPolicy(retrain_policy.percentage.threshold,
                                             retrain_policy.percentage.onlyPositives)
        else:
            raise NotImplementedError('unknown retrain policy: {}'.format(json_format.MessageToJson(retrain_policy)))

    def _get_selector(self, selector: SelectorConfig) -> Selector:
        if selector.HasField('topk'):
            return TopKSelector(selector.topk.k, selector.topk.batchSize,
                                self._get_reexamination_strategy(selector.topk.reexaminationStrategy))
        elif selector.HasField('threshold'):
            return ThresholdSelector(selector.threshold.threshold,
                                     self._get_reexamination_strategy(selector.threshold.reexaminationStrategy))
        else:
            raise NotImplementedError('unknown selector: {}'.format(json_format.MessageToJson(selector)))

    def _get_reexamination_strategy(self, reexamination_strategy: ReexaminationStrategyConfig) -> ReexaminationStrategy:
        if reexamination_strategy.HasField('none'):
            return NoReexaminationStrategy()
        elif reexamination_strategy.HasField('top'):
            return TopReexaminationStrategy(reexamination_strategy.top.k)
        elif reexamination_strategy.HasField('full'):
            return FullReexaminationStrategy()
        else:
            raise NotImplementedError(
                'unknown reexamination strategy: {}'.format(json_format.MessageToJson(reexamination_strategy)))

    def _get_svm_trainer(self, context: ModelTrainerContext, search_id: SearchId, trainer_index: int,
                         config: SVMConfig) -> SVMTrainerBase:
        feature_extractor = config.featureExtractor
        probability = config.probability
        linear_only = config.linearOnly
        if config.mode is SVMMode.MASTER_ONLY:
            return SVMTrainer(context, feature_extractor, self._feature_cache, probability,
                              linear_only)
        elif config.mode is SVMMode.DISTRIBUTED:
            return DistributedSVMTrainer(context, feature_extractor, self._feature_cache, probability,
                                         linear_only, search_id, trainer_index)
        elif config.mode is SVMMode.ENSEMBLE:
            if not config.probability:
                raise NotImplementedError('Probability must be enabled when using ensemble SVM trainer')

            return EnsembleSVMTrainer(context, feature_extractor, self._feature_cache, linear_only,
                                      search_id, trainer_index)
        else:
            raise NotImplementedError('unknown svm mode: {}'.format(config.mode))

    def _get_retriever(self, dataset: Dataset) -> Retriever:
        if dataset.HasField('directory'):
            return DirectoryRetriever(dataset.directory)
        else:
            raise NotImplementedError('unknown dataset: {}'.format(json_format.MessageToJson(dataset)))
