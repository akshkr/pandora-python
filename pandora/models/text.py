from .base import Model


class TextModel(Model):
    def __init__(self):
        super(TextModel, self).__init__()

    def add_preprocessor(self, preprocessor, column=None):
        self.preprocessing_steps.append({'preprocessor': preprocessor, 'column': column})
