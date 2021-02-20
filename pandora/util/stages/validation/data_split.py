from pandora.reference.validation import CrossValType


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
