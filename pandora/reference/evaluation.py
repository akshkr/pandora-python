from enum import Enum
from sklearn.metrics import mean_squared_error


class EvaluationMetrics(Enum):
    EVAL_METRICS_ALIAS = {
        'mse': mean_squared_error
    }
