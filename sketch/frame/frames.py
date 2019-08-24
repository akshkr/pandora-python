import pandas as pd
import warnings


class Canvas:
	
	def __init__(self, train, test, target_column=None, valid_frac=0.25):
		"""
		
		:param train:
		:param test:
		:param target_column:
		:param valid_frac:
		"""
		
		# Train and test data provided by user
		self._train = train.sample(frac=1)
		self._test = test
		
		# check if target column user input else autodetect
		if target_column is None:
			self._target_column = self._auto_detect_target()
		else:
			self._target_column = target_column
		
		# Get train size and validation size
		self._train_size = len(self._train)
		self._valid_size = self._train_size * valid_frac

		# Join data and separate target
		self._data, self._target = self._join_df()
		
		del self._train, self._test, self._target_column
	
	def _auto_detect_target(self):
		"""
		
		:return:
		"""
		print(f'Auto Detecting Target...')
		target_column = list(set(self._train.columns) - set(self._test.columns))
		
		if len(target_column) > 1:
			raise ValueError(f'Found more than one target : {target_column}')
		
		else:
			return target_column[0]
	
	def _join_df(self):
		"""

		:return:
		"""
		
		common_columns = list(set(self._train.columns) & set(self._test.columns))
		
		return pd.concat([self._train[common_columns], self._test[common_columns]], axis=0), self._train[self._target_column]
	
	@property
	def train(self):
		"""
		
		:return:
		"""
		return self._data.iloc[:self._train_size].iloc[:-1*self._valid_size]
	
	@property
	def y_train(self):
		"""

		:return:
		"""
		return self._target.iloc[:-1 * self._valid_size]
	
	@property
	def valid(self):
		"""
		
		:return:
		"""
		return self._data.iloc[:self._train_size].iloc[-1*self._valid_size:]
	
	@property
	def y_valid(self):
		"""

		:return:
		"""
		return self._target.iloc[-1 * self._valid_size:]
	
	@property
	def test(self):
		"""
		
		:return:
		"""
		return self._data.iloc[self._train_size:]
	
	@property
	def data(self):
		"""
		
		:return:
		"""
		return self._data.iloc[:self._train_size]
	
	@property
	def target(self):
		"""

		:return:
		"""
		return self._target
	
	@property
	def dataframe(self):
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
		from sketch.strokes.mathematics import min_max_scale
		
		for col in columns:
			self._data[col] = min_max_scale(self._data, col)
	
	def balance(self, target, bal_frac=0.33, validation=False):
		"""

		:param target:
		:param bal_frac:
		:param validation:
		:return:
		"""
		from sketch.strokes.df_handler import make_balanced_df
		
		if validation:
			# Concat validation train data and target to balance
			data_to_balance = pd.concat([self.train, self.y_train], axis=1)
		else:
			# Concat total train data and target
			warnings.warn(f'The validation data is lost!!!')
			data_to_balace = pd.concat([self.data, self.target], axis=1)
			
		balanced_df = make_balanced_df(data_to_balance, target, bal_frac)
		
		balanced_target = balanced_df[target]
		retained_target = retained_df[target]
		balanced_df.drop(target, axis=1, inplace=True)
		retained_df.drop(target, axis=1, inplace=True)
		
		return balanced_df, retained_df, balanced_target, retained_target
