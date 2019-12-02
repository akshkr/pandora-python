from sketch.frame.statistical import STFrame
from sketch.frame.datatype import DTFrame


class OPFrame(STFrame, DTFrame):
	"""
	Frame to handle the Data Processing operations
	
	Parameters
	----------
	data : DataFrame
		The input is the DataFrame on which the operations are to be done
	"""
	
	def __init__(self, data):
		object.__setattr__(self, "_data", data)
		
		STFrame.__init__(self, data)
		DTFrame.__init__(self, data)
	
	@property
	def data(self):
		"""
		Return the complete DataFrame

		Returns
		-------
		The complete modified DataFrame
		"""
		
		return self._data
