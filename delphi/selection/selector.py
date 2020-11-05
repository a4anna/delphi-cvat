from abc import ABCMeta, abstractmethod
from typing import Optional

from delphi.model import Model
from delphi.result_provider import ResultProvider
from delphi.selection.selector_stats import SelectorStats


class Selector(metaclass=ABCMeta):

    @abstractmethod
    def add_result(self, result: ResultProvider) -> None:
        pass

    @abstractmethod
    def finish(self) -> None:
        pass

    @abstractmethod
    def get_result(self) -> Optional[ResultProvider]:
        pass

    @abstractmethod
    def new_model(self, model: Optional[Model]) -> None:
        pass

    @abstractmethod
    def get_stats(self) -> SelectorStats:
        pass
