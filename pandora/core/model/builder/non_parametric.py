from pandora.core.model.estimators.parameters import PARAMETER_ALIAS
from pandora.util.stages.validation import PARAMETER_SEARCH_ALIAS
from pandora.core.model.estimators import ESTIMATOR_ALIAS
from .base import ModelBuilder

import numpy as np


class NonParametricModelBuilder(ModelBuilder):
    """
    Parametric Model Builder

    This class is used cross validate and get optimal parameters
    For the given model, search method, and dataset

    Parameters
    ----------
    model : str
        Estimator abbreviation
        The abbreviations used are in ESTIMATOR_CLASS variable
    search : str
        Search method abbreviation
        The abbreviations used are in SEARCH_MODEL_ALIAS variable
    """
    def __init__(self, model, search='random', params=None):
        self.model = model
        self.cv_method = search
        self.params = params

    def _init_params(self):
        if not self.params:
            self.params = PARAMETER_ALIAS.get(self.model, None)
        if isinstance(self.model, str):
            self.model = ESTIMATOR_ALIAS.get(self.model, None)

    def build(self, features, target):
        """
        Builds the Parametric model with optimised hyper-parameters

        Parameters
        ----------
        features : np.ndarray
        target : np.ndarray

        Returns
        -------
            Estimator and a dictionary of optimal Hyper-parameters
        """
        search_func = PARAMETER_SEARCH_ALIAS.get(self.cv_method, None)
        self._init_params()

        return self.model, search_func(self.model, self.params, features, target)
