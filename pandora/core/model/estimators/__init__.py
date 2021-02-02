from xgboost import XGBRegressor, XGBClassifier
from sklearn.ensemble import BaggingClassifier

ESTIMATOR_ALIAS = {
    'xgbr': XGBRegressor,
    'xgbc': XGBClassifier,

}

__all__ = ['ESTIMATOR_ALIAS']
