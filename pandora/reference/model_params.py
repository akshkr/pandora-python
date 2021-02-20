from enum import Enum


class ModelParameters(Enum):
    PARAMS_XGBR = {
        "learning_rate": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
        "max_depth": [3, 4, 5, 6, 8, 10, 12, 15],
        "min_child_weight": [1, 3, 5, 7],
        "gamma": [0.0, 0.1, 0.2, 0.3, 0.4],
        "colsample_bytree": [0.3, 0.4, 0.5, 0.7],
    }

    PARAMETER_ALIAS = {
        'xgbr': PARAMS_XGBR
    }
