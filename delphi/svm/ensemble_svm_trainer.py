import pickle
from pathlib import Path
from typing import Union

import torch.multiprocessing as mp
from google.protobuf import json_format
from google.protobuf.any_pb2 import Any
from logzero import logger
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC

from delphi.context.model_trainer_context import ModelTrainerContext
from delphi.model import Model
from delphi.model_trainer import TrainingStyle, DataRequirement
from delphi.proto.internal_pb2 import InternalMessage
from delphi.proto.delphi_pb2 import SearchId
from delphi.proto.svm_trainer_pb2 import SetTrainResult, SVMTrainerMessage
from delphi.svm.feature_cache import FeatureCache
from delphi.svm.pretrained_voting_classifier import PretrainedVotingClassifier
from delphi.svm.svm_model import SVMModel
from delphi.svm.svm_trainer_base import SVMTrainerBase, C_VALUES, GAMMA_VALUES


class EnsembleSVMTrainer(SVMTrainerBase):

    def __init__(self, context: ModelTrainerContext, feature_extractor: str, cache: FeatureCache,
                 linear_only: bool, search_id: SearchId, trainer_index: int):
        super().__init__(context, feature_extractor, cache, True)

        self._search_id = search_id
        self._trainer_index = trainer_index

        self._param_grid = [{'C': C_VALUES, 'kernel': ['linear']}]

        if not linear_only:
            self._param_grid.append({'C': C_VALUES, 'gamma': GAMMA_VALUES, 'kernel': ['rbf']})

        self._train_condition = mp.Condition()
        self._train_id = None
        self._train_results = {}

    @property
    def data_requirement(self) -> DataRequirement:
        return DataRequirement.DISTRIBUTED_POSITIVES

    @property
    def training_style(self) -> TrainingStyle:
        return TrainingStyle.DISTRIBUTED

    def train_model(self, train_dir: Path) -> Model:
        version = self.get_new_version()

        best_model = self.get_best_model(train_dir, self._param_grid)[0]

        if self.context.node_index == 0:
            with self._train_condition:
                self._add_train_results(version, best_model)
                while True:
                    if len(self._train_results[version]) == len(self.context.nodes):
                        results = self._train_results[version]
                        del self._train_results[version]
                        break
                    self._train_condition.wait()

            return SVMModel(PretrainedVotingClassifier(results), version, self.feature_provider, self.probability)
        else:
            message = Any()
            message.Pack(
                SVMTrainerMessage(setTrainResult=SetTrainResult(version=version, model=pickle.dumps(best_model))))
            self.context.nodes[0].internal.MessageInternal(
                InternalMessage(searchId=self._search_id, trainerIndex=self._trainer_index, message=message))

            return SVMModel(best_model, version, self.feature_provider, self.probability)

    def message_internal(self, request: Any) -> Any:
        message = SVMTrainerMessage()
        request.Unpack(message)

        if message.HasField('setTrainResult'):
            return self._set_train_result(message.setTrainResult)
        else:
            logger.error('Unrecognized message type {}'.format(json_format.MessageToJson(request)))

    def _set_train_result(self, request: SetTrainResult) -> Any:
        assert self.context.node_index == 0
        model = pickle.loads(request.model)

        with self._train_condition:
            self._add_train_results(request.version, model)
            remaining = len(self.context.nodes) - len(self._train_results[request.version])
            logger.info(
                'Received training result for model version {} ({} remaining)'.format(request.version, remaining))
            if remaining == 0:
                self._train_condition.notify_all()

        return Any()

    def _add_train_results(self, model_version: int, model: Union[SVC, CalibratedClassifierCV]) -> None:
        if model_version not in self._train_results:
            self._train_results[model_version] = []
        self._train_results[model_version].append(model)
