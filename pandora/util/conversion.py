from scipy.sparse.csr import csr_matrix
import pandas as pd
import numpy as np


def convert_to_numpy(list_to_convert):
    """
    Convert items in a iterator to numpy array

    Args:
        list_to_convert (iterator): List of different DataType

    Returns:
        List with elements converted to numpy array
    """

    return_list = list()
    for i in list_to_convert:
        if isinstance(i, np.ndarray):
            if i.ndim == 1:
                return_list.append(i.reshape(i.shape[0], 1))
            else:
                return_list.append(i)

        elif isinstance(i, csr_matrix):
            return_list.append(i.toarray())

        else:
            raise TypeError(f'Unsupported Datatype: {type(i)}')

    return return_list


def get_values(data):
    if isinstance(data, pd.Series) or isinstance(data, pd.DataFrame):
        return data.values
    else:
        return data
