from abc import ABCMeta, abstractmethod


class BaseTransformer(metaclass=ABCMeta):
    """
    Base Transformer class

    Every child class must implement fit_transform function
    """

    @abstractmethod
    def fit_transform(self, **kwargs):
        raise NotImplementedError
