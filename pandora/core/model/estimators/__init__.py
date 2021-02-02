from xgboost import XGBRegressor, XGBClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression

ESTIMATOR_ALIAS = {
    'xgbr': XGBRegressor,
    'xgbc': XGBClassifier,

}

__all__ = ['ESTIMATOR_ALIAS']
