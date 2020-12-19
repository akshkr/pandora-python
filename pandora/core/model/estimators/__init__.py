from xgboost import XGBRegressor, XGBClassifier


ESTIMATOR_ALIAS = {
    'xgbr': XGBRegressor,
    'xgbc': XGBClassifier
}

__all__ = ['ESTIMATOR_ALIAS']
