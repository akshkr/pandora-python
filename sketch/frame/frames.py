import pandas as pd
import warnings


class Canvas:
	
	def __init__(self, train, test):
		self._train = train
		self._test = test
		
		self._train_size = len(self._train)
		self._data, self._target = self._join_df(self._train, self._test)
		
		del self._train, self._test
	
	@staticmethod
	def _join_df(train, test):
		"""

		:param train:
		:param test:
		:return:
		"""
		
		common_columns = list(set(train.columns) & set(test.columns))
		target_column = list(set(train.columns) - set(test.columns))
		
		if len(target_column) > 1:
			warnings.warn(f'More than one target column detected : {target_column}')
		
		return pd.concat([train[common_columns], test[common_columns]], axis=0), train[target_column]
	
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
			
	def min_max(self, columns):
		"""
		
		:param columns:
		:return:
		"""
		from sketch.strokes.mathematics import min_max_scale
		
		for col in columns:
			min_max_scale(self._data, col)
			
		return self
	
	def balance(self, target, frac=0.33):
		"""
		
		:param target:
		:param frac:
		:return:
		"""
		from sketch.strokes.df_handler import make_balanced_df
		
		data_to_balance = pd.concat([self.train, self._target], axis=1)
		test = self.test
		balanced_df = make_balanced_df(data_to_balance, target, frac)
		self._target = balanced_df[target]
		
		self._train_size = len(balanced_df)
		balanced_df.drop(target, axis=1, inplace=True)
		self._data = balanced_df.append(test)
		
		return self
	
	@property
	def train(self):
		"""
		
		:return:
		"""
		return self._data.iloc[:self._train_size, :]
	
	@property
	def test(self):
		"""
		
		:return:
		"""
		return self._data.iloc[self._train_size:, :]
	
	@property
	def dataframe(self):
		"""
		
		:return:
		"""
		
		return self._data
	
	@property
	def target(self):
		"""
		
		:return:
		"""
		return self._target
