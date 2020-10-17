from .base import Model


class TextModel(Model):
    def __init__(self):
        super(TextModel, self).__init__()

    def add_preprocessor(self, function, column=None):
        if not isinstance(column, str):
            raise ValueError

        self.preprocessing_steps.append((function, column))
