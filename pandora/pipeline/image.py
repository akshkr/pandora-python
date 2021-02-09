from .handler.generator import compile_generator
from .base import BasePipeline


class ImagePipeline(BasePipeline):
    def __init__(self, model=None):
        model = 'image' if model is None else model
        super().__init__(model)
        self._generator = None

    def add_image_generator(self, method, directory, dataframe=None, **generator_args):
        """
        Adds Image generator to the pipeline

        Parameters
        ----------
        method : str ('dir' or 'df')
            Method to generate images, either from DataFrame or Directory
        directory : str
            Directory of images
        dataframe : DataFrame
            DataFrame with labels
        generator_args
            Arguments for generator
        """
        if method not in ['dir', 'df']:
            raise ValueError(f'Invalid input. Enter "dir" or "df".')

        self._template.add_generator_params(method=method, directory=directory, dataframe=dataframe, **generator_args)

    def add_augmentation(self, **augmentation_params):
        """
        Adds augmentation to the Image pipeline

        Parameters
        ----------
        augmentation_params
            Augmentation Parameters
        """
        self._template.add_augmentation_params(**augmentation_params)

    def set_processor(self, *args, **kwargs):
        pass

    def enable_cv(self, validation_split=None):
        """
        Enables Cross-validation

        Parameters
        ----------
        validation_split : float
            Validation ratio
        """
        self._template.add_cross_validation(validation_split=validation_split)

    def compile(self, transformer=None, estimator=None, **estimator_args):
        """
        Adds Transformer and Estimator to the template

        Parameters
        ----------
        transformer
        estimator : object
            Model to Estimate predictions
        estimator_args
            arguments for Estimator
        """
        self._template.add_generator(
            compile_generator(self._template.augmentation_params)
        )
        self._template.add_transformer(transformer)
        self._template.add_estimator(estimator, **estimator_args)

    def run(self, *args, **kwargs):
        pass

    def predict(self, *args, **kwargs):
        pass
