from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold


DATA_SPLIT_ALIAS = {
    'k': KFold,
    'stratified': StratifiedKFold,
    'group': GroupKFold
}


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
        splitter = DATA_SPLIT_ALIAS[splitter]

    if callable(splitter) and hasattr(splitter, 'split'):
        splitter = splitter(n_splits=n_splits, shuffle=True)

    return splitter.split(features, target)
