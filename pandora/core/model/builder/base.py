from abc import ABCMeta, abstractmethod


class BaseModelBuilder(metaclass=ABCMeta):
    """
    Base Model Builder class

    Mostly used to tune parameters of a model
    """
    @abstractmethod
    def build(self, features, target):
        raise NotImplementedError
