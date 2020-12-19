from pandora.core.preprocessor.base import Transformer
from pandora.factory import StatisticalPreprocessor

import numpy as np


class StatisticalMeasure(Transformer):

    def fit_transform(self, features):
        preprocessor = StatisticalPreprocessor(features)
        return np.hstack[preprocessor.central_tendency(), preprocessor.moments()]
