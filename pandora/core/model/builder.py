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
    def __init__(self, model, search='random'):
        self.model = model
        self.cv_method = search

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
        parameters = PARAMETER_ALIAS.get(self.model, None)

        return model, search_func(model, parameters, features, target)
