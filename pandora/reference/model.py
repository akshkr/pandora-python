from enum import Enum
from xgboost import XGBRegressor, XGBClassifier


class Estimators(Enum):
    ESTIMATOR_ALIAS = {
        'xgbr': XGBRegressor,
        'xgbc': XGBClassifier,
    }
