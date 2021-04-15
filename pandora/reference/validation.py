from enum import Enum
from sklearn.model_selection import KFold, StratifiedKFold
from pandora.util.stages.validation import random_search, grid_search


class CrossValType(Enum):
    DATA_SPLIT_ALIAS = {
        'KFold': KFold,
        'StratifiedKFold': StratifiedKFold,
    }


class SearchType(Enum):
    PARAMETER_SEARCH_ALIAS = {
        'random': random_search,
        'grid': grid_search
    }
