from sketch.frame.operation import OPFrame
import pandas as pd
import warnings


class Canvas(OPFrame):
	
	def __init__(self, train, test, target_column_name=None):
		"""
		
		:param train:
		:param test:
		:param target_column_name:
		"""
		
		# Initialize train
		self._train = train.sample(frac=1)
		
		# Initialize test and target column
		if test is not None:
			self._test = test
		
			# check if target column user input else autodetect
			if target_column_name is None:
				self._target_column_name = self._auto_detect_target()
			else:
				self._target_column_name = target_column_name
		else:
			self._test = None
			if target_column_name is not None:
				self._target_column_name = target_column_name
			else:
				warnings.warn('No Target column defined')
				self._target_column_name = None
		
		# Get train size and validation size
		self._train_size = len(self._train)

		# Join data and separate target
		data, target = self._join_df()
		
		OPFrame.__init__(self, data, target)
		del self._train, self._test
	
	def _auto_detect_target(self):
		"""
		
		:return:
		"""
		print(f'Auto Detecting Target...')
		target_column = list(set(self._train.columns) - set(self._test.columns))
		
		if len(target_column) > 1:
			raise ValueError(f'Found more than one target : {target_column}')
		else:
			print(f'Target Column detected : {target_column[0]}')
			return target_column[0]
	
	def _join_df(self):
		"""

		:return:
		"""
		# If no target column
		if self._target_column_name is None:
			return self._train, None
		
		# Handling cases with test data
		if self._test is not None:
			common_columns = list(set(self._train.columns) & set(self._test.columns))
		
			return pd.concat([self._train[common_columns], self._test[common_columns]], axis=0),\
				self._train[self._target_column_name]
		else:
			# When no test data given
			return self._train.drop(self._target_column_name), self._train[self._target_column_name]
	
	@property
	def train(self):
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
	def test(self):
		"""
		
		:return:
		"""
		if self._test is None:
			print(f'No Test data detected')
			return None
		return self._data.iloc[self._train_size:]
	
	@property
	def dataframe(self):
		"""
		
		:return:
		"""
		
		return self._data
	
	def balance(self, bal_frac=0.33, validation=False):
		"""

		:param bal_frac:
		:param validation:
		:return:
		"""
		return NotImplementedError(f'This function is not yet implemented')
		"""
		from sketch.util.df_handler import make_balanced_df
		
		if validation:
			# Concat validation train data and target to balance
			size_idx = len(self.train)
			data_to_balance = pd.concat([self.train, self.y_train], axis=1)
			
		else:
			# Concat total train data and target
			warnings.warn(f'The validation data is lost!!!')
			size_idx = len(self.train)
			data_to_balance = pd.concat([self.train, self.target], axis=1)
			self._valid_size = 0
		
		# Balancing Dataframe
		balanced_df = make_balanced_df(data_to_balance, self._target_column_name, bal_frac)
		balanced_target = balanced_df[self._target_column_name]
		
		# Reinitialize object vars
		self._train_size = self._train_size + (len(balanced_df) - size_idx)
		self._data = pd.concat([balanced_df.drop(self._target_column_name, axis=1), self._data.iloc[size_idx:]], axis=0).reset_index(drop=True)
		self._target = pd.concat([balanced_target, self._target.iloc[size_idx:]]).reset_index(drop=True)
		"""
