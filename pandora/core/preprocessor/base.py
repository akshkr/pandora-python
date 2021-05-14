from abc import ABCMeta, abstractmethod


class BaseTransform(metaclass=ABCMeta):
    """
    Base Transform class

    Used to transform data
    """
    @abstractmethod
    def fit_transform(self, features):
        raise NotImplementedError
