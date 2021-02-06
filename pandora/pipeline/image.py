from pandora.factory import get_template
from .base import Pipeline


class ImagePipeline(Pipeline):
    def __init__(self, model=None):
        model = 'image' if model is None else model
        super().__init__(model)

    def enable_cv(self, *args, **kwargs):
        pass

    def disable_cv(self):
        pass

    def run(self, *args, **kwargs):
        pass

    def predict(self, *args, **kwargs):
        pass
