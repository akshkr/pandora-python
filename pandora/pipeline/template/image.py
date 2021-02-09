from .base import BaseTemplate


class ImageTemplate(BaseTemplate):
    """
    Template for Image pipeline

    This template is used for Image dataset based pipeline
    """
    def __init__(self):
        super().__init__()
        self.augmentation_params = None

    def add_preprocessor(self, preprocessor):
        """
        Adds preprocessor to the template

        Parameters
        ----------
        preprocessor
            Image preprocessor to be used on the input image
        """
        self.preprocessing_steps.append(preprocessor)

    def add_augmentation(self, **augmentation_params):
        """
        Adds Image Augmentation parameters to the template

        Parameters
        ----------
        augmentation_params
            Augmentation parameters
        """
        self.augmentation_params = augmentation_params
