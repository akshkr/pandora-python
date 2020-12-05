import numpy as np


def fit_transform(operator, vector):
    # If the operator is function then map
    # Else fit_transform
    if callable(operator):
        return np.array(list(map(operator, vector)))

    operator_obj = operator
    if hasattr(operator_obj, 'fit_transform'):
        values = operator_obj.fit_transform(vector)
        return values

    raise TypeError(f'Unsupported operator. Pass function or class with "fit_transform" attribute.')


def transform(operator, vector):
    if callable(operator):
        return np.array(list(map(operator, vector)))

    operator_obj = operator
    if hasattr(operator_obj, 'transform'):
        values = operator_obj.transform(vector)
        return values

    raise TypeError(f'Unsupported operator. Pass function or class with "transform" attribute.')
