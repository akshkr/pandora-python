from .base import BaseTemplate


class ImageTemplate(BaseTemplate):
    def __init__(self):
        super().__init__()

    def add_preprocessor(self, preprocessor):
        self.preprocessing_steps.append(preprocessor)
