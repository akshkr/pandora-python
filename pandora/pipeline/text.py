from ..util.datatype import convert_to_numpy
from ..util.process import parallelize
from ..factory import get_model
from .base import Pipeline
from .handler import *
import numpy as np


class TextPipeline(Pipeline):
    def __init__(self, model=None):
        model = 'text' if model is None else model
        self.model = get_model(model)

    def _extract_feature_array(self, data):
        preprocessors = [x['preprocessor'] for x in self.model.preprocessing_steps]
        features = [x['column'] for x in self.model.preprocessing_steps]
        features = [data[col] if col is not None else data for col in features]

        return preprocessors, features

    def add(self, preprocessor, **kwargs):
        self.model.add_preprocessor(preprocessor, **kwargs)

    def compile(self, transformer, estimator):
        self.model.add_transformer(transformer)
        self.model.add_estimator(estimator)

    def run(self, features, target):
        if self.model.preprocessing_steps:
            preprocessors, features = self._extract_feature_array(features)
            features, models = parallelize(
                handle_train_preprocessor,
                zip(preprocessors, features),
                n_jobs=1
            )

            features = convert_to_numpy(features)
            features = np.hstack(features)

        if self.model.transformer:
            pass

        if self.model.estimator:
            handle_train_estimator(self.model.estimator, features, target)

        return features

    def predict(self, *args, **kwargs):
        pass
