from pandora.pipeline.template.base import Template


class ImageTemplate(Template):
    def __init__(self):
        super().__init__()

    def add_preprocessor(self, preprocessor):
        self.preprocessing_steps.append(preprocessor)
