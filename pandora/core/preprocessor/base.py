from abc import ABCMeta, abstractmethod


class BaseTransformer(metaclass=ABCMeta):
    """
    Base Transformer class

    Used to transform data
    """
    @abstractmethod
    def fit_transform(self, **kwargs):
        raise NotImplementedError
