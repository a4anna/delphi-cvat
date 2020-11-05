from abc import ABCMeta, abstractmethod
from typing import List, Optional, Dict


class FeatureCache(metaclass=ABCMeta):

    @abstractmethod
    def get(self, key: str) -> Optional[List[float]]:
        pass

    @abstractmethod
    def put(self, values: Dict[str, List[float]], expire: bool) -> None:
        pass
