from pandora.factory import get_template
from .base import BasePipeline


class ImagePipeline(BasePipeline):
    def __init__(self, model=None):
        model = 'image' if model is None else model
        super().__init__(model)

    def set_processor(self, *args, **kwargs):
        pass

    def enable_cv(self, *args, **kwargs):
        pass

    def disable_cv(self):
        pass

    def run(self, *args, **kwargs):
        pass

    def predict(self, *args, **kwargs):
        pass
