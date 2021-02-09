from .base import BaseTemplate


class ImageTemplate(BaseTemplate):
    """
    Template for Image pipeline

    This template is used for Image dataset based pipeline
    """
    def __init__(self):
        super().__init__()
        self.augmentation_params = None
        self.generator_params = None
        self.generator = None

    def add_augmentation_params(self, **augmentation_params):
        """
        Adds Image Augmentation parameters to the template

        Parameters
        ----------
        augmentation_params
            Augmentation parameters
        """
        self.augmentation_params = augmentation_params

    def add_generator_params(self, method=None, directory=None, dataframe=None, **generator_args):
        """
        Adds Generator parameters to the template

        Parameters
        ----------
        method : str ('dir' or 'df')
            Method to generate images, either from DataFrame or Directory
        directory : str
            Directory of images
        dataframe : DataFrame
            DataFrame with labels
        generator_args
            Generator arguments
        """
        self.generator_params = {
            'method': method,
            'directory': directory,
            'dataframe': dataframe,
            'generator_params': generator_args
        }

    def add_generator(self, generator):
        """
        Adds generator to the template

        Parameters
        ----------
        generator
            Generate object
        """
        self.generator = generator

    def add_preprocessor(self, preprocessor):
        """
        Adds preprocessor to the template

        Parameters
        ----------
        preprocessor
            Image preprocessor to be used on the input image
        """
        self.preprocessing_steps.append(preprocessor)
