from .base import Template


class ImageTemplate(Template):
    def __init__(self):
        super(ImageTemplate, self).__init__()

    def add_preprocessor(self, preprocessor):
        self.preprocessing_steps.append(preprocessor)
