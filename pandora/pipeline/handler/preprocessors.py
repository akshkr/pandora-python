import numpy as np

from pandora.util.stages.transformation import fit_transform, transform
from pandora.util.conversion import get_values, convert_to_numpy


def handle_train_preprocessor(preprocessor, feature):
    """
    Runs the Preprocessor on the given Feature(s)

    Parameters
    ----------
    preprocessor : function or object or list
        Preprocessor to be operated on feature
    feature
        feature vector
    Returns
    -------
        transformed values
    """
    # preprocessor_list = list()

    # If No preprocessor is passed return the raw values
    if preprocessor is None:
        return [get_values(feature)]

    # If input preprocessors are a list of preprocessor
    # Run N-1 preprocessor and append the list of trained preprocessor
    if isinstance(preprocessor, list):
        for i in preprocessor[:-1]:
            feature = fit_transform(i, feature)

        preprocessor = preprocessor[-1]

    # If input is one preprocessor/ last preprocessor of the list
    # Get the transformed values and return
    transformed_values = fit_transform(preprocessor, feature)

    return [transformed_values]


def handle_test_preprocessor(preprocessor, feature):
    """
    Runs the pre-fitted preprocessor on feature

    Parameters
    ----------
    preprocessor : function or object or list
        pre-fitted Preprocessor to be operated on feature
    feature
        feature vector
    Returns
    -------
        transformed values
    """
    if preprocessor is None:
        return [get_values(feature)]

    if isinstance(preprocessor, list):
        for i in preprocessor[:-1]:
            feature = transform(i, feature)
        preprocessor = preprocessor[-1]

    transformed_values = transform(preprocessor, feature)

    return [transformed_values]


def hstack_from_list(data):
    """
    Stack Data Horizontally

    Stacks the list of data exhausted by multiprocessor.

    Parameters
    ----------
    data
        Input Data

    Returns
    -------
        Horizontally stacked data
    """
    data = tuple(*data)
    data = convert_to_numpy(data)
    data = np.hstack(data)

    return data
