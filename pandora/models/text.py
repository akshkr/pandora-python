from .base import Model


class TextModel(Model):
    def __init__(self):
        super(TextModel, self).__init__()

    def add_preprocessor(self, preprocessor, column=None):
        self.preprocessing_steps.append({'preprocessor': preprocessor, 'column': column})

    def update_preprocessor_models(self, model_list):
        if len(self.preprocessing_steps) != len(model_list):
            raise ValueError(f'Length {len(self.preprocessing_steps)} and {len(model_list)} does not match!')

        for idx in range(len(self.preprocessing_steps)):
            self.preprocessing_steps[idx].update({'preprocessor': model_list[idx]})
