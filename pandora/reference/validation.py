from sklearn.model_selection import KFold, StratifiedKFold
from enum import Enum


class CrossValType(Enum):
    DATA_SPLIT_ALIAS = {
        'KFold': KFold,
        'StratifiedKFold': StratifiedKFold,
    }


class SearchType(Enum):
    from pandora.util.stages.validation.parameter_search import random_search, grid_search

    PARAMETER_SEARCH_ALIAS = {
        'random': random_search,
        'grid': grid_search
    }
