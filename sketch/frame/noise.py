from sketch.core.elementary import count_remove


class NFrame:
	"""
	Frame to handle noise
	
	Parameters
	----------
	data : DataFrame
		The input is the DataFrame on which the operations are to be done
	"""
	
	def __init__(self, data):
		object.__setattr__(self, "_data", data)

	def reset_values(self, columns, min_count=10):
		"""
		Remove values which have less occurrence in a data column
		
		Parameters
		----------
		columns : List of strings
			removes values with less occurrence
		min_count : int
			values less than min_count are removed
		"""
		
		for column in columns:
			
			self._data[column] = count_remove(self._data, column, min_count)
