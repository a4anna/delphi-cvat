from delphi.retrain.retrain_policy import RetrainPolicy


class AbsoluteThresholdPolicy(RetrainPolicy):

    def __init__(self, threshold: int, only_positives: bool):
        super().__init__()
        self._threshold = threshold
        self._only_positives = only_positives
        self._new_examples = 0

    def update(self, new_positives: int, new_negatives: int) -> None:
        self._new_examples += new_positives

        if not self._only_positives:
            self._new_examples += new_negatives

    def should_retrain(self) -> bool:
        return self._new_examples >= self._threshold

    def reset(self) -> None:
        self._new_examples = 0
