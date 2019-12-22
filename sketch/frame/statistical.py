from sketch.core.encode import dummy_encoder, label_encoder, frequency_encoder
import pandas as pd
import warnings


class STFrame:
	"""
	Frame to handle Statistical Data Processing part like encoding and scaling etc.
	
	Parameters
	----------
	data : DataFrame
		The input is the DataFrame on which the operations are to be done
	"""
	
	def __init__(self, data):
		object.__setattr__(self, "_data", data)
	
	def one_hot_encode(self, columns, drop=True):
		"""
		encode the given columns list using One hot encoding

		Parameters
		----------
		columns : List of strings
			Encodes all the columns names in the list using
			one hot encoding
		drop : boolean
			If true drops the original columns after encoding
		"""
		self._data = dummy_encoder(self._data, columns, drop)
	
	def label_encode(self, columns=None, label='unknown', all_cols=False, verbose=False):
		"""
		Encodes the given columns list using Label encoding

		Parameters
		----------
		columns : List of strings, default None
			List of columns to be encoded
		label : String, default unknown
			Label to fill in place of Nan (missing values)
		all_cols : boolean, default False
			If True encodes every compatible columns
		verbose : boolean, default False
			If True prints the columns being encoded
		"""
		# TODO: Exclude target from being encoded
		if columns is None and not all_cols:
			print(f'Pass "all=True" to encode every object type column')
			return
		
		# Making a list of all columns with object data type to encode
		if all_cols:
			warnings.warn(f'Using all columns with Object Data Type!')
			columns = list()
			for col in self._data.columns:
				if self._data[col].dtype == 'O':
					columns.append(col)
		
		if verbose:
			print(f'Columns being encoded : {columns}')
		
		self._data = label_encoder(self._data, columns, label)
	
	def frequency_encode(self, columns, drop=False):
		"""
		Encodes columns using Frequency encoding

		Parameters
		----------
		columns : List of string
			List of columns to encode
		drop : boolean, default False
			If True drops the original columns after encoding
		"""
		self._data = frequency_encoder(self._data, columns, drop)
		
	def map_dict(self, columns, map_dictionary):
		"""
		Maps the column using dictionary

		Parameters
		----------
		columns : List of strings
			List of columns to be mapped
		map_dictionary : Dictionary
			dictionary used for mapping data in the given columns
		"""
		for col in columns:
			self._data[col] = self._data[col].map(map_dictionary)
	
	def min_max(self, columns):
		"""
		Normalises columns using Min-max scaling

		Parameters
		----------
		columns : List of strings
			List of columns to be normalised
		"""
		from sketch.core.scale import min_max_scale
		
		for col in columns:
			self._data[col] = min_max_scale(self._data, col)
