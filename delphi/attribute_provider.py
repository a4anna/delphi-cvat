from abc import ABCMeta, abstractmethod
from typing import Mapping

# Must be picklable
class AttributeProvider(metaclass=ABCMeta):

    @abstractmethod
    def get(self) -> Mapping[str, bytes]:
        pass


class SimpleAttributeProvider(AttributeProvider):

    def __init__(self, attributes: Mapping[str, bytes]):
        self._attributes = attributes

    def get(self) -> Mapping[str, bytes]:
        return self._attributes

