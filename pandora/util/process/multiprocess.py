from joblib import Parallel, delayed


def parallelize(function, arguments, n_jobs):
    return Parallel(n_jobs=n_jobs, backend='multiprocessing')(delayed(function)(*i) for i in arguments)
