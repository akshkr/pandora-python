from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import numpy as np


def random_search(model, params, features, target):
    """
    Random Search for Hyper-parameters

    Parameters
    ----------
    model : class
        Estimator class
    params : dict
        dictionary of hyper-parameters to be searched
    features : np.ndarray
    target : np.ndarray

    Returns
    -------
        Dictionary of optimal Hyper-parameters
    """
    cross_validator = RandomizedSearchCV(model(), params, refit=False)
    return cross_validator.fit(features, target).best_params_


def grid_search(model, params, features, target):
    """
    Grid Search for Hyper-parameters

    Parameters
    ----------
    model : class
        Estimator class
    params : dict
        dictionary of hyper-parameters to be searched
    features : np.ndarray
    target : np.ndarray

    Returns
    -------
        Dictionary of optimal Hyper-parameters
    """
    cross_validator = GridSearchCV(model(), params, refit=False)
    return cross_validator.fit(features, target).best_params_
