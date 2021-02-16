from pandora.util.stages.estimation import fit, predict, fit_all, predict_all, fit_gen
from pandora.util.stages.validation import base_n_fold_splitter
from pandora.util.stages.evaluation import EVAL_METRICS_ALIAS
from copy import deepcopy


def handle_train_estimator(estimator, features=None, target=None, generator=None, **estimator_args):
    """
    Fits Estimator with given features and target

    Parameters
    ----------
    estimator : object
    features
        Input features/ independent variable
    target
        Input target/ Dependent variable
    generator
        Input data generator
    estimator_args
        Arguments to be passes in Estimator fit function

    Returns
    -------
        Trained model
    """
    if features is not None and target is not None:
        return fit(estimator, features, target, **estimator_args)
    elif generator is not None:
        return fit_gen(estimator, generator.generate(subset='training'), **estimator_args)


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


def handle_cv(cv_params, estimator, features, target):
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
    """
    # Get index list of cross-validation
    split_index = base_n_fold_splitter(cv_params['method'], features, target, cv_params['n_split'])
    train_index, test_index = list(zip(*split_index))

    # Make copies of estimator to multiprocess
    estimator_list = [deepcopy(estimator) for i in range(cv_params['n_split'])]

    estimator_list = fit_all(estimator_list, features, target, n_jobs=cv_params['n_jobs'], index=train_index)
    predictions = predict_all(estimator_list, features, n_jobs=cv_params['n_jobs'], index=test_index)

    for t in list(zip(test_index, predictions)):
        for metric in cv_params['metrics']:
            print(f'{metric}: {EVAL_METRICS_ALIAS[metric](target[t[0]], t[1])}')
