from abc import abstractmethod


class Model:

    def __init__(self):
        self.preprocessing_steps = list()
        self.transformer = None
        self.evaluator = None

    @abstractmethod
    def add_preprocessor(self, *args, **kwargs):
        raise NotImplementedError

    def add_transformer(self, transformer):
        self.transformer = transformer

    def add_model(self, evaluator=None):
        self.evaluator = evaluator
