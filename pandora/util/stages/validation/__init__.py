from .parameter_search import random_search, grid_search

SEARCH_MODEL_ALIAS = {
    'random': random_search,
    'grid': grid_search
}
