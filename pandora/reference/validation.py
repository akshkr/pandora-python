from enum import Enum
from sklearn.model_selection import KFold, StratifiedKFold
from pandora.util.stages.validation import random_search, grid_search


class CrossValType(Enum):
    """
    Cross Validation data split strategies
    """
    DATA_SPLIT_ALIAS = {
        'KFold': KFold,
        'StratifiedKFold': StratifiedKFold,
    }


class SearchType(Enum):
    """
    Optimal parameter search strategies

    These function when passed model, params, features, target
    return the optimal parameters
    """
    PARAMETER_SEARCH_ALIAS = {
        'random': random_search,
        'grid': grid_search
    }
