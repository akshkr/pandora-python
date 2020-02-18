from scipy.sparse.csr import csr_matrix
import numpy as np


def convert_to_numpy(list_to_convert):
	"""
	Convert items in a list to numpy array
	
	Args:
		list_to_convert (list): List of different DataType

	Returns:
		List with elements converted to numpy array
	"""
	
	return_list = list()
	for i in list_to_convert:
		if isinstance(i, np.ndarray):
			return_list.append(i.reshape(i.shape[0], 1))
			
		elif isinstance(i, csr_matrix):
			return_list.append(i.toarray())
	
	return return_list
