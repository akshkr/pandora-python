from joblib import Parallel, delayed


def parallelize(function, arguments, n_jobs, multi_return=True):
    """
    Runs a function to a list of arguments in parallel

    Parameters
    ----------
    function : callable
        function to run in parallel
    arguments : list
        list of arguments
    n_jobs : int
        Number of process to run in parallel
    multi_return : bool
        True in function returns a multi-values list

    Returns
    -------
        zipped output values
    """
    values = Parallel(n_jobs=n_jobs, backend='multiprocessing')(delayed(function)(*i) for i in arguments)
    if multi_return:
        return zip(*values)
    else:
        return values
