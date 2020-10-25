from abc import abstractmethod


class Model:

    def __init__(self):
        self.preprocessing_steps = list()
        self.transformer = None
        self.estimator = None

    @abstractmethod
    def add_preprocessor(self, *args, **kwargs):
        raise NotImplementedError

    def add_transformer(self, transformer):
        self.transformer = transformer

    def add_estimator(self, estimator):
        self.estimator = estimator
