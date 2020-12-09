from ...util.stages.validation import SEARCH_MODEL_ALIAS
from ..model.parameters import ESTIMATOR_ALIAS
from .parameters import PARAMETER_ALIAS
from ..model.base import ModelBuilder

import numpy as np


class ParametricModelBuilder(ModelBuilder):
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
        search_func = SEARCH_MODEL_ALIAS.get(self.cv_method, None)
        model = ESTIMATOR_ALIAS.get(self.model, None)
        if not self.params:
            self.params = PARAMETER_ALIAS.get(self.model, None)

        return model, search_func(model, self.params, features, target)
