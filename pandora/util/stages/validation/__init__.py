from .parameter_search import random_search, grid_search
from .data_split import base_n_fold_splitter, DATA_SPLIT_ALIAS


PARAMETER_SEARCH_ALIAS = {
    'random': random_search,
    'grid': grid_search
}
