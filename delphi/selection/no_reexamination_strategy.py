import queue
from typing import List

from delphi.model import Model
from delphi.selection.reexamination_strategy import ReexaminationStrategy


class NoReexaminationStrategy(ReexaminationStrategy):

    @property
    def revisits_old_results(self) -> bool:
        return False

    def get_new_queues(self, model: Model, old_queues: List[queue.PriorityQueue]) -> List[queue.PriorityQueue]:
        return [queue.PriorityQueue()]
