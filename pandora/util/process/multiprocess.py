from joblib import Parallel, delayed


def parallelize(function, arguments, n_jobs):
    values = Parallel(n_jobs=n_jobs, backend='multiprocessing')(delayed(function)(*i) for i in arguments)
    print(values)
    return zip(*values)
