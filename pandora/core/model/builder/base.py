from abc import ABCMeta, abstractmethod


class ModelBuilder(metaclass=ABCMeta):

    @abstractmethod
    def build(self, features, target):
        raise NotImplementedError
