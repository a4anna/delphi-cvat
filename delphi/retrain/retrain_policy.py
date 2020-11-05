from abc import ABCMeta, abstractmethod


class RetrainPolicy(metaclass=ABCMeta):

    @abstractmethod
    def update(self, new_positives: int, new_negatives: int) -> None:
        pass

    @abstractmethod
    def should_retrain(self) -> bool:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass
