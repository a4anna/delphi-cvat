import os
import threading
import time
from typing import Dict, Optional, List

from google.protobuf.wrappers_pb2 import BytesValue
from logzero import logger

from delphi.condition.model_condition import ModelCondition
from delphi.delphi_stub import DelphiStub
from delphi.model_trainer import ModelTrainer
from delphi.proto.delphi_pb2 import ModelStats


class BandwidthCondition(ModelCondition):

    def __init__(self, node_index: int, nodes: List[DelphiStub], threshold_mbps: float, refresh_seconds: int,
                 trainer: ModelTrainer):
        super().__init__()
        self._node_index = node_index
        self._nodes = nodes
        self._threshold_mbps = threshold_mbps
        self._refresh_seconds = refresh_seconds
        self._trainer = trainer

        self._latest_rate = 0
        self._rate_lock = threading.Lock()
        self._abort_event = threading.Event()

        if self._node_index != 0:
            threading.Thread(target=self._check_bandwidth, name='check-bandwidth').start()

    def is_satisfied(self, example_counts: Dict[str, int], last_statistics: Optional[ModelStats]) -> bool:
        if len(self._nodes) == 1:
            return True  # Only one node running a search - no data transfer happening at all

        with self._rate_lock:
            rate = self._latest_rate

        return rate > self._threshold_mbps

    def close(self) -> None:
        self._abort_event.set()

    @property
    def trainer(self) -> ModelTrainer:
        return self._trainer

    def _check_bandwidth(self):
        while not self._abort_event.set():
            try:
                rate = 0
                request = BytesValue(value=bytearray(os.urandom(1024 * 1024)))
                for node in self._nodes[1:]:
                    request_start = time.time()
                    response_message = node.internal.CheckBandwidth(request)
                    request_time = time.time() - request_start
                    rate += ((len(request.value) + len(response_message.value)) / 1024 / 1024 / request_time)

                rate /= (len(self._nodes) - 1)  # Average bandwidth measurement across nodes
                with self._rate_lock:
                    self._latest_rate = rate
            except Exception as e:
                logger.exception(e)

            time.sleep(self._refresh_seconds)
