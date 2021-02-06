from pandora.util.stages.estimation import fit, predict
from pandora.util.stages.validation import base_n_fold_splitter


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


def handle_cv_train(cv_params, estimator, features, target):
    """
    Handles Cross-validation training

    Parameters
    ----------
    cv_params : dict
        Cross-validation parameters dictionary
    estimator : object
        Estimator object
    features : array or DataFrame
        Features to be trained upon
    target : array or DataFrame
        Target to be trained upon

    Returns
    -------

    """
    split_index = base_n_fold_splitter(cv_params['method'], features, target, cv_params['n_split'])



