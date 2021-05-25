from copy import deepcopy

from pandora.util.stages.estimation import fit, predict, fit_all, predict_all, fit_gen
from pandora.reference.evaluation import EvaluationMetrics
from pandora.reference.validation import CrossValType


def handle_train_estimator(estimator, features=None, target=None, generator=None, **estimator_args):
    """
    Fits Estimator with given features and target or generator

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

    raise ValueError('No data passed. Pass features/target or generator.')


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


def base_n_fold_splitter(splitter, features, target, n_splits=4):
    """
    N Fold split for cross validation

    Parameters
    ----------
    splitter : callable or object
        Splitter object or class to perform split
    features : np.ndarray or pd.DataFrame
        Features to be split
    target : np.ndarray or pd.DataFrame
        Target to be split
    n_splits : int
        Number of folds to be split

    Returns
    -------
        Split index object
    """
    if isinstance(splitter, str):
        splitter = CrossValType.DATA_SPLIT_ALIAS.value[splitter]

    if callable(splitter) and hasattr(splitter, 'split'):
        splitter = splitter(n_splits=n_splits, shuffle=True)

    return splitter.split(features, target)


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

    estimator_list = fit_all(
        estimator_list, features, target, n_jobs=cv_params['n_jobs'], index=train_index
        )
    predictions = predict_all(
        estimator_list, features, n_jobs=cv_params['n_jobs'], index=test_index
        )

    for eval_data in list(zip(test_index, predictions)):
        for metric in cv_params['metrics']:
            print(f'{metric}: {EvaluationMetrics.EVAL_METRICS_ALIAS.value[metric](target[eval_data[0]], eval_data[1])}')
