from ..factory import get_model
from .base import Pipeline


class Text(Pipeline):
    def __init__(self, model=None):
        model = 'text' if model is None else model
        self.model = get_model(model)

    def add(self, function, column):
        self.model.add_preprocessor(function, column)

    def run(self, transformer, estimator):
        self.model.add_transformer(transformer)
        self.model.add_estimator(estimator)

    def predict(self, *args, **kwargs):
        pass
