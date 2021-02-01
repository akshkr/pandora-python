from pandora.util.stages.estimation import fit, predict


def handle_train_estimator(estimator, features, target, **estimator_args):
    """
    Fits Estimator with given features and target

    Parameters
    ----------
    estimator : object
    features
        Input features/ independent variable
    target
        Input target/ Dependent variable
    estimator_args
        Arguments to be passes in Estimator fit function

    Returns
    -------
        Trained model
    """
    return fit(estimator, features, target, **estimator_args)


def handle_test_estimator(estimator, features):
    """
    Predicts the target with given features and estimator

    Parameters
    ----------
    estimator : object
    features
        Input features/ independent variable

    Returns
    -------
        predicted output
    """
    return predict(estimator, features)
