import numpy as np


def fit_transform(operator, vector):
    """
    This function is used to handle a preprocessor on a feature.
    The preprocessor can be a function of a class with "fit_transform"
    member function.

    Parameters
    ----------
    operator: callable
        Preprocessor
    vector
        Feature

    Returns
    -------
        Preprocessed values
    """
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
    """
    This function is used to handle a preprocessor on a feature.
    The preprocessor can be a function of a class with "transform"
    member function.
    This is mostly used when the operator is fit on some previous data

    Parameters
    ----------
    operator: callable
        Preprocessor
    vector
        Feature

    Returns
    -------
        Preprocessed values
    """
    if callable(operator):
        return np.array(list(map(operator, vector)))

    operator_obj = operator
    if hasattr(operator_obj, 'transform'):
        values = operator_obj.transform(vector)
        return values

    raise TypeError(f'Unsupported operator. Pass function or class with "transform" attribute.')
