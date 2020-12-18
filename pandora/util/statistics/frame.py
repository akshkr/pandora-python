from scipy import stats
import numpy as np


class Frame:

    def __init__(self, frame):
        self.frame = frame

    def central_tendency(self):
        sum_ = np.sum(self.frame, axis=1)
        mean_ = np.mean(self.frame, axis=1)
        median_ = np.median(self.frame, axis=1)

        return np.array([sum_, mean_, median_]).T

    def moments(self):
        std_ = np.std(self.frame, axis=1)
        skew_ = stats.skew(self.frame, axis=1)
        kurtosis_ = stats.kurtosis(self.frame, axis=1)

        return np.array([std_, skew_, kurtosis_]).T
