from scipy import stats
import numpy as np


class Frame:
    """
    Statistical operation class for all frame types(DataFrame, Numpy array)
    """

    def __init__(self, frame):
        self.frame = frame

    def central_tendency(self):
        """
        Calculate central tendency for given dataset

        Returns
        -------
            Numpy array with sum, mean, median
        """
        sum_ = np.sum(self.frame, axis=1)
        mean_ = np.mean(self.frame, axis=1)
        median_ = np.median(self.frame, axis=1)

        return np.array([sum_, mean_, median_]).T

    def moments(self):
        """
        Calculate Moments for given dataset

        Returns
        -------
            Numpy array with standard deviation, skewness and kurtosis
        """
        std_ = np.std(self.frame, axis=1)
        skew_ = stats.skew(self.frame, axis=1)
        kurtosis_ = stats.kurtosis(self.frame, axis=1)

        return np.array([std_, skew_, kurtosis_]).T
