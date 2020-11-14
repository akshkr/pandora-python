from abc import abstractmethod


class Template:

    def __init__(self):
        self.preprocessing_steps = list()
        self.transformer = None
        self.estimator = None
        self.estimator_args = None

    @abstractmethod
    def add_preprocessor(self, *args, **kwargs):
        raise NotImplementedError

    def add_transformer(self, transformer):
        self.transformer = transformer

    def add_estimator(self, estimator, **kwargs):
        self.estimator = estimator
        self.estimator_args = kwargs

