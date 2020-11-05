import queue
import threading
from abc import abstractmethod
from typing import Optional

from delphi.model import Model
from delphi.result_provider import ResultProvider
from delphi.selection.selector import Selector


class SelectorBase(Selector):

    def __init__(self):
        self.result_queue = queue.Queue(maxsize=100)
        self.stats_lock = threading.Lock()
        self.items_processed = 0

        self._model_lock = threading.Lock()
        self._model_present = False

        self._finish_event = threading.Event()

    @abstractmethod
    def add_result_inner(self, result: ResultProvider) -> None:
        pass

    @abstractmethod
    def new_model_inner(self, model: Optional[Model]) -> None:
        pass

    def add_result(self, result: ResultProvider) -> None:
        with self._model_lock:
            model_present = self._model_present

        if not model_present:
            self.result_queue.put(result)
        else:
            self.add_result_inner(result)

        with self.stats_lock:
            self.items_processed += 1

    def new_model(self, model: Optional[Model]) -> None:
        with self._model_lock:
            self._model_present = model is not None

        self.new_model_inner(model)

    def finish(self) -> None:
        self._finish_event.set()
        self.result_queue.put(None)

    def get_result(self) -> Optional[ResultProvider]:
        while True:
            try:
                return self.result_queue.get(timeout=10)
            except queue.Empty:
                if self._finish_event.is_set():
                    return None
