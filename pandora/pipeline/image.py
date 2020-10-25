from ..factory import get_model
from .base import Pipeline


class Image(Pipeline):
    def __init__(self, model=None):
        model = 'image' if model is None else model
        self.model = get_model(model)

    def add(self, preprocessor):
        self.model.add_preprocessor(preprocessor)

    def compile(self, transformer, estimator):
        self.model.add_transformer(transformer)
        self.model.add_estimator(estimator)

    def run(self, *args, **kwargs):
        pass

    def predict(self, *args, **kwargs):
        pass
