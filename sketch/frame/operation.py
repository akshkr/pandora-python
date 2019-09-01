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
	
	def one_hot_encode(self, feature, drop=True):
		"""

		:param feature:
		:param drop:
		:return:
		"""
		if drop:
			self._data = pd.concat([pd.get_dummies(self._data[feature]), self._data.drop(feature, axis=1)], axis=1)
		else:
			self._data = pd.concat([pd.get_dummies(self._data[feature]), self._data], axis=1)
	
	def label_encode(self, features):
		"""

		:param features:
		:return:
		"""
		from sklearn.preprocessing import LabelEncoder
		label_encoder = LabelEncoder()
		
		for col in features:
			self._data[col] = label_encoder.fit_transform(self._data[col])
	
	def min_max(self, columns):
		"""

		:param columns:
		:return:
		"""
		from sketch.util.mathematics import min_max_scale
		
		for col in columns:
			self._data[col] = min_max_scale(self._data, col)
