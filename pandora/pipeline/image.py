from ..factory import get_model
from .base import Pipeline


class Image(Pipeline):
    def __init__(self, model=None):
        model = 'image' if model is None else model
        self.model = get_model(model)

    def add(self, function):
        self.model.add_preprocessor(function)

    def run(self, transformer, estimator):
        self.model.add_transformer(transformer)
        self.model.add_estimator(estimator)

    def predict(self, *args, **kwargs):
        pass
