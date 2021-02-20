from sklearn.metrics import mean_squared_error
from enum import Enum


class EvaluationMetrics(Enum):
    EVAL_METRICS_ALIAS = {
        'mse': mean_squared_error
    }
