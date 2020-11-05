import queue
from typing import List

from delphi.model import Model
from delphi.selection.reexamination_strategy import ReexaminationStrategy


class TopReexaminationStrategy(ReexaminationStrategy):

    def __init__(self, k: int):
        self._k = k

    @property
    def revisits_old_results(self) -> bool:
        return True

    def get_new_queues(self, model: Model, old_queues: List[queue.PriorityQueue]) -> List[queue.PriorityQueue]:
        new_queue = queue.PriorityQueue()

        to_reexamine = []
        for priority_queue in old_queues:
            for _ in range(self._k):
                try:
                    to_reexamine.append(priority_queue.get_nowait()[1])
                except queue.Empty:
                    break

        for result in model.infer(to_reexamine):
            new_queue.put((-result.score, result.id, result))

        return old_queues + [new_queue]
