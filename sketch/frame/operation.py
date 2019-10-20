import pandas as pd
import warnings


class OPFrame:
	"""
	Frame to handle just the Data Processing part like encoding, scaling etc
	
	Parameters
	----------
	
	data : DataFrame
		The input is the DataFrame on which the operations are to be done
	"""
	
	def __init__(self, data):
		object.__setattr__(self, "_data", data)
	
	@property
	def data(self):
		"""
		Return the complete DataFrame

		Returns
		-------
		The complete modified DataFrame
		"""
		
		return self._data
	
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
		encoded_data = pd.get_dummies(self._data[columns])
		
		if drop:
			self._data = pd.concat([self._data.drop(columns=columns), encoded_data], axis=1)
		else:
			self._data = pd.concat([self._data, encoded_data], axis=1)
	
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
		
		from sklearn.preprocessing import LabelEncoder
		label_encoder = LabelEncoder()
		
		# Handling Nan values
		print(f'filling NaN values with label "{label}"')
		self._data[columns].fillna(label, inplace=True)
		self._data[columns] = self._data[columns].astype(str)
		
		# Encoding using Label Encoder
		for col in columns:
			self._data[col] = label_encoder.fit_transform(self._data[col])
			
		# Converting the data type to categorical
		self._data[columns] = self._data[columns].astype('category')
			
	def frequency_encode(self, columns, verbose=False, drop=False):
		"""
		Encodes columns using Frequency encoding
		
		Parameters
		----------
		columns : List of string
			List of columns to encode
		verbose : boolean, default False
			If True prints the value assigned to a label
		drop : boolean, default False
			If True drops the original columns after encoding
		"""
		
		for col in columns:
			col_encoded = self._data[col].value_counts().to_dict()
			self._data[f'{col}_freq_enc'] = self._data[col].map(col_encoded)
			
			if verbose:
				print(f'Encoded columns {col} as : {col_encoded}')
				
		if drop:
			self._data.drop(columns=columns, inplace=True)
				
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
		from sketch.util.mathematics import min_max_scale
		
		for col in columns:
			self._data[col] = min_max_scale(self._data, col)
