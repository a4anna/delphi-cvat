from abc import ABCMeta, abstractmethod
from collections import Sized
from typing import Iterable

from delphi.object_provider import ObjectProvider
from delphi.proto.delphi_pb2 import DelphiObject
from delphi.retrieval.retriever_stats import RetrieverStats


class Retriever(metaclass=ABCMeta):

    @abstractmethod
    def start(self) -> None:
        pass

    # @abstractmethod
    # def stop(self) -> None:
    #     pass

    @abstractmethod
    def get_objects(self) -> Iterable[ObjectProvider]:
        pass

    # @abstractmethod
    # def get_object(self, object_id: str, attributes: Sized) -> DelphiObject:
    #     pass

    # @abstractmethod
    # def get_stats(self) -> RetrieverStats:
    #     pass
