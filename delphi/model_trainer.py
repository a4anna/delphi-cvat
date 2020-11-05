from abc import ABCMeta, abstractmethod
from enum import Enum, auto, unique
from pathlib import Path
import threading

from google.protobuf.any_pb2 import Any

from delphi.model import Model


@unique
class DataRequirement(Enum):
    MASTER_ONLY = 0
    DISTRIBUTED_POSITIVES = 1
    DISTRIBUTED_FULL = 2


class TrainingStyle(Enum):
    MASTER_ONLY = auto()
    DISTRIBUTED = auto()


class ModelTrainer(metaclass=ABCMeta):

    @property
    @abstractmethod
    def data_requirement(self) -> DataRequirement:
        pass

    @property
    @abstractmethod
    def training_style(self) -> TrainingStyle:
        pass

    @property
    @abstractmethod
    def should_sync_model(self) -> bool:
        pass

    @abstractmethod
    def load_from_file(self, model_version: int, file: bytes) -> Model:
        pass

    @abstractmethod
    def train_model(self, train_dir: Path) -> Model:
        pass

    @abstractmethod
    def message_internal(self, request: Any) -> Any:
        pass


class ModelTrainerBase(ModelTrainer):

    def __init__(self):
        self._latest_version = 0
        self._version_lock = threading.Lock()

    def get_new_version(self):
        with self._version_lock:
            self._latest_version += 1
            version = self._latest_version
        return version
    
