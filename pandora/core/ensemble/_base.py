from abc import ABCMeta, abstractmethod


class BaseEnsemble(metaclass=ABCMeta):
    """
    Base Class for Ensemble

    Parameters
    ----------
    estimator : object
        Estimator
    n_estimators : int
        Number of Estimators
    n_jobs : int
        Number of parallel process
    """

    def __init__(self, estimator=None, n_estimators=None, n_jobs=None):
        self._n_jobs = n_jobs
        self.estimator = estimator
        self.n_estimator = n_estimators

    def set_params(self, estimators):
        self.estimator = estimators

    @abstractmethod
    def fit(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def predict(self, *args, **kwargs):
        raise NotImplementedError
