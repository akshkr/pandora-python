from abc import ABCMeta, abstractmethod


class Transformer(metaclass=ABCMeta):

    @abstractmethod
    def fit_transform(self, **kwargs):
        raise NotImplementedError
