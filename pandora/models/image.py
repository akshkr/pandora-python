from .base import Model


class ImageModel(Model):
    def __init__(self):
        super(ImageModel, self).__init__()

    def add_preprocessor(self, function):
        self.preprocessing_steps.append(function)
