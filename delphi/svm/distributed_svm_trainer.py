import json
import pickle
import threading
import time
from itertools import cycle
from pathlib import Path
from typing import List, Dict

import torch.multiprocessing as mp
from google.protobuf import json_format
from google.protobuf.any_pb2 import Any
from logzero import logger

from delphi.context.model_trainer_context import ModelTrainerContext
from delphi.model import Model
from delphi.model_trainer import TrainingStyle, DataRequirement
from delphi.proto.internal_pb2 import InternalMessage
from delphi.proto.delphi_pb2 import SearchId
from delphi.proto.svm_trainer_pb2 import SetTrainResult, SVMTrainerMessage, SetParamGrid
from delphi.svm.feature_cache import FeatureCache
from delphi.svm.svm_model import SVMModel
from delphi.svm.svm_trainer_base import SVMTrainerBase
from delphi.utils import log_exceptions


class DistributedSVMTrainer(SVMTrainerBase):

    def __init__(self, context: ModelTrainerContext, feature_extractor: str, cache: FeatureCache,
                 probability: bool, linear_only: bool, search_id: SearchId, trainer_index: int):
        super().__init__(context, feature_extractor, cache, probability)

        self._linear_only = linear_only
        self._search_id = search_id
        self._trainer_index = trainer_index

        self._param_grid = []
        self._param_grid_lock = threading.Lock()

        self._train_condition = mp.Condition()
        self._train_id = None
        self._train_results = {}

        self._param_grid_event = threading.Event()

        if self.context.node_index == 0:
            threading.Thread(target=self._assign_param_grid, name='set-param-grid').start()

    @property
    def data_requirement(self) -> DataRequirement:
        return DataRequirement.DISTRIBUTED_FULL

    @property
    def training_style(self) -> TrainingStyle:
        return TrainingStyle.DISTRIBUTED

    def train_model(self, train_dir: Path) -> Model:
        self._param_grid_event.wait()
        version = self.get_new_version()

        with self._param_grid_lock:
            param_grid = self._param_grid

        best_model = self.get_best_model(train_dir, param_grid)

        if self.context.node_index == 0:
            with self._train_condition:
                self._add_train_results(version, best_model)
                while True:
                    if len(self._train_results[version]) == len(self.context.nodes):
                        results = self._train_results[version]
                        del self._train_results[version]
                        break
                    self._train_condition.wait()

            model, params, score = max(results, key=lambda item: item[2])
            logger.info('Best parameters found across all nodes: {}'.format(params))
            return SVMModel(model, version, self.feature_provider, self.probability)
        else:
            message = Any()
            message.Pack(SVMTrainerMessage(setTrainResult=SetTrainResult(version=version,
                                                                         params=json.dumps(best_model[1]),
                                                                         score=best_model[2],
                                                                         model=pickle.dumps(best_model[0]))))
            self.context.nodes[0].internal.MessageInternal(
                InternalMessage(searchId=self._search_id, trainerIndex=self._trainer_index, message=message))
            return SVMModel(best_model[0], version, self.feature_provider, self.probability)

    def message_internal(self, request: Any) -> Any:
        message = SVMTrainerMessage()
        request.Unpack(message)

        if message.HasField('setTrainResult'):
            return self._set_train_result(message.setTrainResult)
        elif message.HasField('setParamGrid'):
            return self._set_param_grid(message.setParamGrid)
        else:
            logger.error('Unrecognized message type {}'.format(json_format.MessageToJson(request)))

    def _set_train_result(self, request: SetTrainResult) -> Any:
        assert self.context.node_index == 0
        params = json.loads(request.params)
        model = pickle.loads(request.model)

        with self._train_condition:
            self._add_train_results(request.version, (model, params, request.score))
            remaining = len(self.context.nodes) - len(self._train_results[request.version])
            logger.info(
                'Received training result for model version {} ({} remaining)'.format(request.version, remaining))
            if remaining == 0:
                self._train_condition.notify_all()

        return Any()

    def _add_train_results(self, model_version: int, train_results: Any) -> None:
        if model_version not in self._train_results:
            self._train_results[model_version] = []
        self._train_results[model_version].append(train_results)

    def _set_param_grid(self, request: SetParamGrid) -> Any:
        assert self.context.node_index != 0

        self._set_param_grid_inner(json.loads(request.grid))

        logger.info('Param grid set from master node')
        self._param_grid_event.set()

        return Any()

    def _set_param_grid_inner(self, param_grid: List[Dict[str, Any]]) -> None:
        with self._param_grid_lock:
            self._param_grid = param_grid

    @log_exceptions
    def _assign_param_grid(self) -> None:
        assert self.context.node_index == 0

        param_grids = []
        for i in range(len(self.context.nodes)):
            param_grids.append([])

        node_assignments = cycle(range(len(self.context.nodes)))

        Cs = [0.01, 0.1, 1, 10]
        gammas = [0.001, 0.01, 0.1, 1, 10]

        for C in Cs:
            param_grids[next(node_assignments)].append({'C': [C], 'kernel': ['linear']})

        if not self._linear_only:
            for C in Cs:
                for gamma in gammas:
                    param_grids[next(node_assignments)].append({'C': [C], 'gamma': [gamma], 'kernel': ['rbf']})

        self._set_param_grid_inner(param_grids[0])

        for i in range(1, len(self.context.nodes)):
            while True:
                try:
                    message = Any()
                    message.Pack(SVMTrainerMessage(setParamGrid=SetParamGrid(grid=json.dumps(param_grids[i]))))
                    self.context.nodes[i].internal.MessageInternal(
                        InternalMessage(searchId=self._search_id, trainerIndex=self._trainer_index, message=message))
                    break
                except Exception as e:
                    logger.warn(
                        'Could not set param grid - node {} might still not be up'.format(self.context.nodes[i].url))
                    logger.exception(e)
                    time.sleep(5)

            logger.info('Set param grid on node {}'.format(self.context.nodes[i].url))

        self._param_grid_event.set()
