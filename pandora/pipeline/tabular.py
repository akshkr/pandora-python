import pandas as pd

from pandora.core.model.builder.base import BaseModelBuilder
from pandora.util.process.multiprocess import parallelize
from pandora.reference.pipeline import PipelineTypes
from pandora.util.callbacks import PipelineCallback

from .handler.preprocessors import (
    handle_train_preprocessor, handle_test_preprocessor, hstack_from_list
    )
from .handler.estimators import (
    handle_train_estimator, handle_test_estimator, handle_cv
    )
from .base import BasePipeline


class TabularPipeline(BasePipeline):
    """
    Pipeline for Composite Dataset

    This is pipeline for tabular dataset.

    Parameters
    ----------
    model : str
        Type of the model
    """
    def __init__(self, model=None):
        model = PipelineTypes.TABULAR.value if model is None else model
        super().__init__(model)

    def _extract_steps_array(self, data):
        """
        Extracts Preprocessors and Data Vectors from template

        Parameters
        ----------
        data
            Input data to extracts feature vectors

        Returns
        -------
            List of Preprocessors, List of feature vectors
        """
        # separate preprocessor_list and features column from preprocessing steps
        preprocessor_list = [x['preprocessor'] for x in self._template.preprocessing_steps]
        features = [x['column'] for x in self._template.preprocessing_steps]

        # Get the column vector or the passed vector according to input preprocessor_list
        features = [
            data[col] if col is not None and isinstance(data, pd.DataFrame)
            else data[:, col] if col is not None
            else data for col in features
        ]

        return preprocessor_list, features

    def enable_cv(self, method, metrics, n_split=4):
        """
        Enables cross-validation

        Parameters
        ----------
        method : str or object
            Method used for cross-validation
        metrics : list of str
            Evaluation metrics to be used
        n_split
            Number of split for training data
        """
        self._template.add_cross_validation(method=method, metrics=metrics, n_split=n_split, n_jobs=self._n_jobs)

    def set_processor(self, n_jobs):
        """
        Sets Processor parameters

        Parameters
        ----------
        n_jobs : int
            Number of jobs to run in parallel
        """
        self._n_jobs = n_jobs

    def run(self, features, target, verbose=1, callbacks=None, retain_data=False):
        """
        Runs the Pipeline on the given input features and target

        Parameters
        ----------
        features
            Input feature(s)
        target
            Target to estimate
        verbose : int
            run verbose
        callbacks : list
            List of callback objects
        retain_data : bool
            Retain features after preprocessing if True
        """
        if not callbacks:
            callbacks = [PipelineCallback()]
            callbacks[0].set_params({'verbose': verbose})

        # Run Preprocessing steps on the input features
        if self._template.preprocessing_steps:
            for c in callbacks:
                c.on_preprocess_begin()
            preprocessor_list, features = self._extract_steps_array(features)

            # parallel preprocessing
            features = parallelize(
                handle_train_preprocessor,
                zip(preprocessor_list, features),
                n_jobs=self._n_jobs
            )

            features = hstack_from_list(features)
            for c in callbacks:
                c.on_preprocess_end()

            del preprocessor_list

        if self._template.transformer:
            pass

        if self._template.estimator:
            for c in callbacks:
                c.on_estimation_begin()

            # Model Builder
            if isinstance(self._template.estimator, BaseModelBuilder):
                estimator_class, estimator_hyper_params = self._template.estimator.build(features, target)
                self._template.estimator = estimator_class(**estimator_hyper_params)

            # Validation
            if self._template.cross_val:
                handle_cv(self._template.cross_val, self._template.estimator, features, target)

            handle_train_estimator(self._template.estimator, features, target, **self._template.estimator_args)
            for c in callbacks:
                c.on_estimation_end()

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
        if self._template.preprocessing_steps:
            preprocessor_list, features = self._extract_steps_array(features)

            features = parallelize(
                handle_test_preprocessor,
                zip(preprocessor_list, features),
                n_jobs=self._n_jobs
            )

            features = hstack_from_list(features)

        if self._template.transformer:
            pass

        if self._template.estimator:
            prediction_values = handle_test_estimator(self._template.estimator, features)

            return prediction_values

        return None
