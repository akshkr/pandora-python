from .base import BasePipeline


class ImagePipeline(BasePipeline):
    def __init__(self, model=None):
        model = 'image' if model is None else model
        super().__init__(model)
        self._generator = None

    def add_image_generator(self, image_generator):
        self._generator = image_generator

    def add_augmentation(self, **augmentation_params):
        """
        Adds augmentation to the Image pipeline

        Parameters
        ----------
        augmentation_params
            Augmentation Parameters
        """

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
