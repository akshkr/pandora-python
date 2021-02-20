from xgboost import XGBRegressor, XGBClassifier
from enum import Enum


class Estimators(Enum):
    ESTIMATOR_ALIAS = {
        'xgbr': XGBRegressor,
        'xgbc': XGBClassifier,
    }
