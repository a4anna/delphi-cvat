from abc import abstractmethod

from torch.utils.tensorboard import SummaryWriter

from delphi.context.context_base import ContextBase


class ModelTrainerContext(ContextBase):

    @property
    @abstractmethod
    def port(self) -> int:
        pass

    @property
    @abstractmethod
    def tb_writer(self) -> SummaryWriter:
        pass
