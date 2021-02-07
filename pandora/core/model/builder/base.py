from abc import ABCMeta, abstractmethod


class BaseModelBuilder(metaclass=ABCMeta):
    """
    Base Model Builder class
    used for hyper-parameter tuning
    """

    @abstractmethod
    def build(self, features, target):
        raise NotImplementedError
