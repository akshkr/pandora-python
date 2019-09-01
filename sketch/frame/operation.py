import pandas as pd


class OPFrame:
	
	def __init__(self, data):
		"""
		
		:param data:
		"""
		
		object.__setattr__(self, "_data", data)
	
	@property
	def data(self):
		"""

		:return:
		"""
		
		return self._data
	
	def one_hot_encode(self, columns, drop=True):
		"""

		:param columns:
		:param drop:
		:return:
		"""
		encoded_data = pd.get_dummies(self._data[columns])
		
		if drop:
			self._data = pd.concat([self._data.drop(columns, axis=1), encoded_data], axis=1)
		else:
			self._data = pd.concat([self._data, encoded_data], axis=1)
	
	def label_encode(self, columns):
		"""

		:param columns:
		:return:
		"""
		from sklearn.preprocessing import LabelEncoder
		label_encoder = LabelEncoder()
		
		for col in columns:
			self._data[col] = label_encoder.fit_transform(self._data[col])
			
	def frequency_encode(self, columns, verbose=False):
		"""
		
		:param columns:
		:return:
		"""
		
		for col in columns:
			col_encoded = self._data[col].value_counts().to_dict()
			self._data[col] = self._data[col].map(col_encoded)
			
			if verbose:
				print(f'Encoded columns {col} as : {col_encoded}')
	
	def min_max(self, columns):
		"""

		:param columns:
		:return:
		"""
		from sketch.util.mathematics import min_max_scale
		
		for col in columns:
			self._data[col] = min_max_scale(self._data, col)
