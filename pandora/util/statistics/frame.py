from scipy import stats
import numpy as np


class Frame:

    @staticmethod
    def central_tendency(frame):
        sum_ = np.sum(frame, axis=1)
        mean_ = np.mean(frame, axis=1)
        median_ = np.median(frame, axis=1)

        return np.array([sum_, mean_, median_]).T

    @staticmethod
    def moments(frame):
        std_ = np.std(frame, axis=1)
        skew_ = stats.skew(frame, axis=1)
        kurtosis_ = stats.kurtosis(frame, axis=1)

        return np.array([std_, skew_, kurtosis_]).T
