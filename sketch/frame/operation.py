from sketch.frame.statistical import STFrame
from sketch.frame.datatype import DTFrame
from sketch.frame.noise import NFrame


class OPFrame(STFrame, DTFrame, NFrame):
	"""
	Frame to handle the Data Processing operations
	
	Parameters
	----------
	data : DataFrame
		The input is the DataFrame on which the operations are to be done
	"""
	
	def __init__(self, data):
		
		STFrame.__init__(self, data)
		DTFrame.__init__(self, data)
		NFrame.__init__(self, data)
	
	@property
	def data(self):
		"""
		Return the complete DataFrame

		Returns
		-------
		The complete modified DataFrame
		"""
		
		return self._data
