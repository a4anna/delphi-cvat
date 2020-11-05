import math
import queue
import threading
from typing import Optional

from delphi.model import Model
from delphi.result_provider import ResultProvider
from delphi.selection.reexamination_strategy import ReexaminationStrategy
from delphi.selection.selector_base import SelectorBase
from delphi.selection.selector_stats import SelectorStats


class TopKSelector(SelectorBase):

    def __init__(self, k: int, batch_size: int, reexamination_strategy: ReexaminationStrategy):
        assert k < batch_size
        super().__init__()

        self._k = k
        self._batch_size = batch_size
        self._reexamination_strategy = reexamination_strategy

        self._priority_queues = [queue.PriorityQueue()]
        self._batch_added = 0
        self._insert_lock = threading.Lock()

    def add_result_inner(self, result: ResultProvider) -> None:
        with self._insert_lock:
            self._priority_queues[-1].put((-result.score, result.id, result))
            self._batch_added += 1
            if self._batch_added == self._batch_size:
                for _ in range(self._k):
                    self.result_queue.put(self._priority_queues[-1].get()[-1])
                self._batch_added = 0

    def new_model_inner(self, model: Optional[Model]) -> None:
        with self._insert_lock:
            if model is not None:
                # add fractional batch before possibly discarding results in old queue
                for _ in range(math.ceil(float(self._k) * self._batch_added / self._batch_size)):
                    self.result_queue.put(self._priority_queues[-1].get()[-1])
                self._priority_queues = self._reexamination_strategy.get_new_queues(model, self._priority_queues)
            else:
                # this is a reset, discard everything
                self._priority_queues = [queue.PriorityQueue()]

            self._batch_added = 0

    def get_stats(self) -> SelectorStats:
        with self.stats_lock:
            items_processed = self.items_processed

        return SelectorStats(items_processed, 0, None, 0)
