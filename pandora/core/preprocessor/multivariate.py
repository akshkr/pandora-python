import numpy as np

from pandora.factory.preprocessor import StatisticalPreprocessor
from pandora.core.preprocessor.base import BaseTransform


class StatisticsMeasure(BaseTransform):
    """
    Calculates Statistical parameters for the input data

    Sum, Mean, Median, Standard deviation, skewness and kurtosis
    is returned by this transformer
    """

    def fit_transform(self, features):
        preprocessor = StatisticalPreprocessor(features)
        return np.hstack([preprocessor.central_tendency(), preprocessor.moments()])
