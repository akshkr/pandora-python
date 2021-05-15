from pandora.util.callbacks import PipelineCallback
from pandora.reference.pipeline import PipelineTypes

from .handler.estimators import handle_train_estimator, handle_test_estimator
from .handler.generator import compile_generator
from .base import BasePipeline


class ImagePipeline(BasePipeline):
    def __init__(self, model=None):
        model = PipelineTypes.IMAGE.value if model is None else model
        super().__init__(model)

    def add_image_generator(self, method, directory, target_size, dataframe=None, **generator_args):
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
        target_size : tuple
            Target image size
        generator_args
            Arguments for generator
        """
        if method not in ['dir', 'df']:
            raise ValueError('Invalid input. Enter "dir" or "df".')

        self._template.add_generator_params(method, directory, target_size, dataframe=dataframe, **generator_args)

    def add_augmentation(self, **augmentation_params):
        """
        Adds augmentation to the Image pipeline

        Parameters
        ----------
        augmentation_params
            Augmentation Parameters
        """
        self._template.add_augmentation_params(**augmentation_params)

    def set_processor(self, n_jobs):
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

    def compile(self, estimator=None, **estimator_args):
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
        if self._template.generator_params is not None:
            self._template.add_generator(
                compile_generator(self._template)
            )

        # Handle cross-validation parameter
        if self._template.cross_val is not None:
            if self._template.generator_params is not None:
                estimator_args['validation_data'] = self._template.generator.generate(subset='validation')
            else:
                estimator_args['validation_split'] = self._template.cross_val['validation_split']

        self._template.add_estimator(estimator, **estimator_args)

    def run(self, features=None, target=None, verbose=1, callbacks=None, retain_data=False):
        """
        Runs the Image pipeline on the given features and target/generator

        Parameters
        ----------
        features
            Input Feature(s)
        target
            Input target
        verbose : bool
            run verbose
        callbacks : list
            List of callbacks
        """
        if not callbacks:
            callbacks = [PipelineCallback()]
            callbacks[0].set_params({'verbose': verbose})

        if self._template.preprocessing_steps:
            pass

        if self._template.estimator:
            for callback in callbacks:
                callback.on_estimation_begin()

            handle_train_estimator(
                self._template.estimator, features, target, self._template.generator,
                **self._template.estimator_args
            )

            for callback in callbacks:
                callback.on_estimation_end()

        if retain_data:
            self._data = features

    def predict(self, features):
        """
        Predicts target of the input features

        Parameters
        ----------
        features
            Input feature(s)

        Returns
        -------
            Predicted Values
        """
        if self._template.estimator:
            prediction_values = handle_test_estimator(self._template.estimator, features)

            return prediction_values

        return None
