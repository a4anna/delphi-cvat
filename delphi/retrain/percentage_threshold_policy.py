from delphi.retrain.retrain_policy import RetrainPolicy


class PercentageThresholdPolicy(RetrainPolicy):

    def __init__(self, threshold: float, only_positives: bool):
        super().__init__()
        self._threshold = threshold
        self._only_positives = only_positives
        self._num_examples = 0
        self._previous_size = 0

    def update(self, new_positives: int, new_negatives: int) -> None:
        self._num_examples += new_positives

        if not self._only_positives:
            self._num_examples += new_negatives

    def should_retrain(self) -> bool:
        return self._num_examples >= (1 + self._threshold) * self._previous_size

    def reset(self) -> None:
        self._previous_size = self._num_examples
