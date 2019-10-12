import pandas as pd
import warnings


class OPFrame:
	
	def __init__(self, data):
		"""
		
		:param data:
		"""
		
		object.__setattr__(self, "_data", data)
	
	@property
	def data(self):
		"""
		Return the complete DataFrame

		:return: data (complete DataFrame)
		"""
		
		return self._data
	
	def one_hot_encode(self, columns, drop=True):
		"""
		encode the given columns list using One hot encoding

		:param columns: List of columns
		:param drop: boolean (If True, drops the columns to be encoded after encoding)
		"""
		encoded_data = pd.get_dummies(self._data[columns])
		
		if drop:
			self._data = pd.concat([self._data.drop(columns, axis=1), encoded_data], axis=1)
		else:
			self._data = pd.concat([self._data, encoded_data], axis=1)
	
	def label_encode(self, columns=None, label='unknown', all_cols=False, verbose=False):
		"""
		Encodes the given columns list using Label encoding

		:param columns: List of columns to be encoded
		:param label: Label to fill in place of Nan (missing values)
		:param all_cols: boolean (If True encodes every compatible columns)
		"""
		if columns is None and not all_cols:
			print(f'Pass "all=True" to encode every object type column')
			return
		
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
		
		print(f'filling NaN values with label "{label}"')
		self._data[columns].fillna(label, inplace=True)
		self._data[columns] = self._data[columns].astype(str)
		
		for col in columns:
			self._data[col] = label_encoder.fit_transform(self._data[col])
			
		self._data[columns] = self._data[columns].astype('category')
			
	def frequency_encode(self, columns, verbose=False):
		"""
		Encodes given list of columns using Frequency encoding
		
		:param columns: List of columns to encode
		:param verbose: boolean (If True prints the value assigned to a label)
		"""
		
		for col in columns:
			col_encoded = self._data[col].value_counts().to_dict()
			self._data[col] = self._data[col].map(col_encoded)
			
			if verbose:
				print(f'Encoded columns {col} as : {col_encoded}')
	
	def min_max(self, columns):
		"""
		Normalises the given column using Min-max scaling

		:param columns: List of columns to be normalised
		"""
		from sketch.util.mathematics import min_max_scale
		
		for col in columns:
			self._data[col] = min_max_scale(self._data, col)
