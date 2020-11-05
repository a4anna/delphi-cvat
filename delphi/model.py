from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Callable, Iterable

from delphi.object_provider import ObjectProvider
from delphi.result_provider import ResultProvider


class Model(metaclass=ABCMeta):

    @property
    @abstractmethod
    def version(self) -> int:
        pass

    @abstractmethod
    def infer(self, requests: Iterable[ObjectProvider]) -> Iterable[ResultProvider]:
        pass

    @abstractmethod
    def infer_dir(self, directory: Path, callback_fn: Callable[[int, float], None]) -> None:
        pass

    @abstractmethod
    def get_bytes(self) -> bytes:
        pass

    @property
    @abstractmethod
    def scores_are_probabilities(self) -> bool:
        pass
