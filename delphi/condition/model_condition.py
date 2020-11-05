from abc import ABCMeta, abstractmethod
from typing import Dict, Optional

from delphi.model_trainer import ModelTrainer
from delphi.proto.delphi_pb2 import ModelStats


class ModelCondition(metaclass=ABCMeta):

    @abstractmethod
    def is_satisfied(self, example_counts: Dict[str, int], last_statistics: Optional[ModelStats]) -> bool:
        pass

    @abstractmethod
    def close(self) -> None:
        pass

    @property
    @abstractmethod
    def trainer(self) -> ModelTrainer:
        pass
