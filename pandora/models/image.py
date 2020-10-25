from .base import Model


class ImageModel(Model):
    def __init__(self):
        super(ImageModel, self).__init__()

    def add_preprocessor(self, preprocessor):
        self.preprocessing_steps.append(preprocessor)
