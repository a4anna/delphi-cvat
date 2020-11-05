from abc import abstractmethod
from pathlib import Path
from typing import List

from delphi.context.context_base import ContextBase
from delphi.model_trainer import ModelTrainer
from delphi.proto.delphi_pb2 import SearchId


class DataManagerContext(ContextBase):

    @property
    @abstractmethod
    def search_id(self) -> SearchId:
        pass

    @property
    @abstractmethod
    def data_dir(self) -> Path:
        pass

    @abstractmethod
    def get_active_trainers(self) -> List[ModelTrainer]:
        pass

    @abstractmethod
    def new_examples_callback(self, new_positives: int, new_negatives: int) -> None:
        pass


