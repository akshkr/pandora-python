class DTFrame:
	"""
	Frame to handle Data Type processing like Date/time, Amount, Location
	
	Parameters
	----------
	data : DataFrame
		The input is the DataFrame on which the operations are to be done
	"""
	
	def __init__(self, data):
		object.__setattr__(self, "_data", data)
