from pandora.reference.model_params import ModelParameters
from pandora.reference.validation import SearchType
from pandora.reference.model import Estimators
from .base import BaseModelBuilder

import numpy as np


class NonParametricModelBuilder(BaseModelBuilder):
    """
    Parametric Model Builder

    Cross validates and get optimal parameters
    for the given model, search method, and dataset

    Parameters
    ----------
    model : str
        Estimator alias
        The alias used are in ESTIMATOR_CLASS variable
    search : str
        Search method alias
        The alias used are in SEARCH_MODEL_ALIAS variable
    params :
    """
    def __init__(self, model, search='random', params=None):
        self.model = model
        self.cv_method = search
        self.params = params

    def _init_params(self):
        if not self.params:
            self.params = ModelParameters.PARAMETER_ALIAS.value.get(self.model, None)
        if isinstance(self.model, str):
            self.model = Estimators.ESTIMATOR_ALIAS.value.get(self.model, None)

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
        search_func = SearchType.PARAMETER_SEARCH_ALIAS.value.get(self.cv_method, None)
        self._init_params()

        return self.model, search_func(self.model, self.params, features, target)
