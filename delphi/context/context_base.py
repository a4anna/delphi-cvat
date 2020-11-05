from abc import ABCMeta, abstractmethod
from typing import List

from delphi.delphi_stub import DelphiStub


class ContextBase(metaclass=ABCMeta):

    @property
    @abstractmethod
    def node_index(self) -> int:
        pass

    @property
    @abstractmethod
    def nodes(self) -> List[DelphiStub]:
        pass
