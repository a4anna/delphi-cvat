import queue
from abc import ABCMeta, abstractmethod
from typing import List

from delphi.model import Model


class ReexaminationStrategy(metaclass=ABCMeta):

    @property
    @abstractmethod
    def revisits_old_results(self) -> bool:
        pass

    @abstractmethod
    def get_new_queues(self, model: Model, old_queues: List[queue.PriorityQueue]) -> List[queue.PriorityQueue]:
        pass
