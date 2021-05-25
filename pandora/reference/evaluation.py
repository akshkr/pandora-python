from enum import Enum
from sklearn.metrics import mean_squared_error


class EvaluationMetrics(Enum):
    """
    Contains the evaluation metrics function

    The function must take true and real output
    and return the score
    """
    EVAL_METRICS_ALIAS = {
        'mse': mean_squared_error
    }
