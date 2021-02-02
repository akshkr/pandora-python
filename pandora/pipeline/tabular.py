from pandora.pipeline.handler.preprocessors import hstack_from_list
from pandora.util.stages.validation import base_n_fold_splitter
from pandora.util.callbacks import PipelineCallback
from pandora.core.model.builder import ModelBuilder
from pandora.util.process import parallelize

from .base import Pipeline
from .handler import *

import pandas as pd


class TabularPipeline(Pipeline):
    """
    Pipeline for Composite Dataset

    This is pipeline for tabular dataset.

    Parameters
    ----------
    model : str
        Type of the model
    """
    def __init__(self, model=None):
        model = 'tabular' if model is None else model
        super().__init__(model)
        self.cv_params = None

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

    def enable_cv(self, method, n_split=4):
        """
        Enables cross-validation

        Parameters
        ----------
        method : str or object
            Method used for cross-validation
        n_split
            Number of split for training data
        """
        self.cv_params = {'method': method, 'n_split': n_split}

    def run(self, features, target, verbose=1, callbacks=None, retain_features=False):
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
        retain_features : bool
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
            if isinstance(self._template.estimator, ModelBuilder):
                estimator_class, estimator_hyper_params = self._template.estimator.build(features, target)
                self._template.estimator = estimator_class(**estimator_hyper_params)

            # Validation
            if self.cv_params:
                base_n_fold_splitter(self.cv_params['method'], features, target, self.cv_params['n_split'])

            handle_train_estimator(self._template.estimator, features, target, **self._template.estimator_args)
            for c in callbacks:
                c.on_estimation_end()

        if retain_features:
            self._features = features

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
