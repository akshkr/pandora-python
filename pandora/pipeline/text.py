from ..util.datatype import convert_to_numpy
from ..util.process import parallelize
from ..factory import get_template
from .base import Pipeline
from .handler import *
import numpy as np


class TextPipeline(Pipeline):
    def __init__(self, model=None):
        model = 'text' if model is None else model
        self.model = get_template(model)

    def _extract_steps_array(self, data):
        # separate preprocessors and features column from preprocessing steps
        preprocessors = [x['preprocessor'] for x in self.model.preprocessing_steps]
        features = [x['column'] for x in self.model.preprocessing_steps]

        # Get the column vector or the passed vector according to input preprocessors
        features = [data[col] if col is not None else data for col in features]

        return preprocessors, features

    def add(self, preprocessor, **kwargs):
        self.model.add_preprocessor(preprocessor, **kwargs)

    def compile(self, transformer=None, estimator=None):
        self.model.add_transformer(transformer)
        self.model.add_estimator(estimator)

    def run(self, features, target):
        if self.model.preprocessing_steps:
            preprocessors, features = self._extract_steps_array(features)
            features = parallelize(
                handle_train_preprocessor,
                zip(preprocessors, features),
                n_jobs=1
            )

            features = tuple(*features)
            features = convert_to_numpy(features)
            features = np.hstack(features)

        if self.model.transformer:
            pass

        if self.model.estimator:
            handle_train_estimator(self.model.estimator, features, target)

    def predict(self, features):
        if self.model.preprocessing_steps:
            preprocessors, features = self._extract_steps_array(features)
            features = parallelize(
                handle_test_preprocessor,
                zip(preprocessors, features),
                n_jobs=1
            )

            features = tuple(*features)
            features = convert_to_numpy(features)
            features = np.hstack(features)

        if self.model.transformer:
            pass

        if self.model.estimator:
            prediction_values = handle_test_estimator(self.model.estimator, features)

            return prediction_values

        return None
