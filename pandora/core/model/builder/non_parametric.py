import numpy as np

from pandora.reference.model_params import ModelParameters
from pandora.reference.validation import SearchType
from pandora.reference.model import Estimators
from pandora.util.handler.error import check_key
from .base import BaseModelBuilder


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
    search : str, optional
        Search method alias
        The alias used are in SEARCH_MODEL_ALIAS variable
    params : dict, optional
        hyper-parameter space of model
        If not passed, ModelParameters.PARAMETER_ALIAS is used
    """
    def __init__(self, model, search='random', params=None):
        self.model = model
        self.cv_method = search
        self.params = params

    def _init_params(self):
        # re-initialize model hyper-parameter space if string is passed
        if self.params is None:
            parameter_dict = ModelParameters.PARAMETER_ALIAS.value
            check_key(self.model, parameter_dict)

            self.params = parameter_dict.get(self.model, None)

        # re-initialize model if only string is passed
        if isinstance(self.model, str):
            estimator_dict = Estimators.ESTIMATOR_ALIAS.value
            check_key(self.model, estimator_dict)

            self.model = estimator_dict.get(self.model, None)

    def build(self, features, target):
        """
        Builds the Parametric model with optimised hyper-parameters

        Parameters
        ----------
        features : np.ndarray
            Features to be used for parameter tuning
        target : np.ndarray
            Target to be used for parameter tuning

        Returns
        -------
            Estimator and a dictionary of optimal Hyper-parameters
        """
        search_func_dict = SearchType.PARAMETER_SEARCH_ALIAS.value
        check_key(self.cv_method, search_func_dict)

        search_func = search_func_dict.get(self.cv_method, None)
        self._init_params()

        return self.model, search_func(self.model, self.params, features, target)
