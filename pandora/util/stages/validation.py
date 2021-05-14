from sklearn.model_selection import RandomizedSearchCV, GridSearchCV


def random_search(model, params, features, target):
    """
    Random Search for Hyper-parameters

    Parameters
    ----------
    model : class
        Estimator class
    params : dict
        dictionary of hyper-parameters to be searched
    features : np.ndarray
    target : np.ndarray

    Returns
    -------
        Dictionary of optimal Hyper-parameters
    """
    cross_validator = RandomizedSearchCV(model(), params, refit=False)
    return cross_validator.fit(features, target).best_params_


def grid_search(model, params, features, target):
    """
    Grid Search for Hyper-parameters

    Parameters
    ----------
    model : class
        Estimator class
    params : dict
        dictionary of hyper-parameters to be searched
    features : np.ndarray
    target : np.ndarray

    Returns
    -------
        Dictionary of optimal Hyper-parameters
    """
    cross_validator = GridSearchCV(model(), params, refit=False)
    return cross_validator.fit(features, target).best_params_


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
    from pandora.reference.validation import CrossValType

    if isinstance(splitter, str):
        splitter = CrossValType.DATA_SPLIT_ALIAS.value[splitter]

    if callable(splitter) and hasattr(splitter, 'split'):
        splitter = splitter(n_splits=n_splits, shuffle=True)

    return splitter.split(features, target)
