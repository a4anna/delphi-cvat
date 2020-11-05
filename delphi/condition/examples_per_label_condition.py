from typing import Dict, Optional

from logzero import logger

from delphi.condition.model_condition import ModelCondition
from delphi.model_trainer import ModelTrainer
from delphi.proto.delphi_pb2 import ModelStats


class ExamplesPerLabelCondition(ModelCondition):

    def __init__(self, count: int, trainer: ModelTrainer):
        super().__init__()
        self._count = count
        self._trainer = trainer

    def is_satisfied(self, example_counts: Dict[str, int], last_statistics: Optional[ModelStats]) -> bool:
        if len(example_counts) < 2:
            logger.info('Less than two label types present ({})'.format(len(example_counts)))
            return False

        for label in example_counts:
            if example_counts[label] < self._count:
                logger.info('Less than {} labels present for label {} (found {})'.format(self._count, label,
                                                                                         example_counts[label]))
                return False

        return True

    def close(self) -> None:
        pass

    @property
    def trainer(self) -> ModelTrainer:
        return self._trainer
