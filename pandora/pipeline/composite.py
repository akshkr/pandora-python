from ..util.conversion import convert_to_numpy
from ..util.process import parallelize
from ..factory import get_template
from .base import Pipeline
from .handler import *
import pandas as pd
import numpy as np


class CompositePipeline(Pipeline):
    """
    Pipeline for Composite Dataset

    This is pipeline for composite dataset.

    Parameters
    ----------
    model : str
        Type of the model

    Attributes
    ----------
    template : object
        Template object encapsulates skeleton of the Pipeline.
        It comprises of preprocessing steps, transformer, estimator
    """
    def __init__(self, model=None):
        model = 'composite' if model is None else model
        self.template = get_template(model)

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
        preprocessor_list = [x['preprocessor'] for x in self.template.preprocessing_steps]
        features = [x['column'] for x in self.template.preprocessing_steps]

        # Get the column vector or the passed vector according to input preprocessor_list
        features = [
            data[col] if col is not None and isinstance(data, pd.DataFrame)
            else data[:, col] if col is not None
            else data for col in features
        ]

        return preprocessor_list, features

    def add(self, preprocessor=None, **kwargs):
        """
        Adds preprocessing Steps to pipeline

        Parameters
        ----------
        preprocessor : object or function  or list
            Preprocessor class or function of list containing Preprocessor
            or functions. These are the function(s) applied to one vector
        kwargs
            column/columns
        """
        self.template.add_preprocessor(preprocessor, **kwargs)

    def compile(self, transformer=None, estimator=None, **kwargs):
        """
        Adds Transformer and Estimator to the template

        Parameters
        ----------
        transformer
        estimator : object
            Model to Estimate predictions
        kwargs
            arguments for Estimator
        """
        self.template.add_transformer(transformer)
        self.template.add_estimator(estimator, **kwargs)

    def run(self, features, target):
        """
        Runs the Pipeline on the given input features and target

        Parameters
        ----------
        features
            Input feature(s)
        target
            Target to estimate
        """
        # Run Preprocessing steps on the input features
        if self.template.preprocessing_steps:
            preprocessor_list, features = self._extract_steps_array(features)
            features = parallelize(
                handle_train_preprocessor,
                zip(preprocessor_list, features),
                n_jobs=1
            )

            features = tuple(*features)
            features = convert_to_numpy(features)
            features = np.hstack(features)

        if self.template.transformer:
            pass

        if self.template.estimator:
            handle_train_estimator(self.template.estimator, features, target, **self.template.estimator_args)

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
        if self.template.preprocessing_steps:
            preprocessor_list, features = self._extract_steps_array(features)
            features = parallelize(
                handle_test_preprocessor,
                zip(preprocessor_list, features),
                n_jobs=1
            )

            features = tuple(*features)
            features = convert_to_numpy(features)
            features = np.hstack(features)

        if self.template.transformer:
            pass

        if self.template.estimator:
            prediction_values = handle_test_estimator(self.template.estimator, features)

            return prediction_values

        return None
