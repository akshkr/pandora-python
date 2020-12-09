from xgboost import XGBRegressor, XGBClassifier
from .tree_model import *


ESTIMATOR_ALIAS = {
    'xgbr': XGBRegressor,
    'xgbc': XGBClassifier
}

PARAMETER_ALIAS = {
    'xgbr': PARAMS_XGBR
}
