from sketch.util.dfhandler import reduce_mem_usage
from sketch.frame.operation import OPFrame
import pandas as pd
import warnings


class Canvas(OPFrame):
	
	def __init__(
		self,
		train,
		test,
		target_column_name,
		reduce_memory
	):
		"""
		
		:param train: Training Data
		:param test: Test Data
		:param target_column_name: The column to be predicted. If not specified tried to detect on its own
		:param reduce_memory: True to reduce the memory taken by DataFrame by reducing the Data Type
		"""
		
		# Shuffle and initialize train
		self._train = train.sample(frac=1)
		
		# Initialize test and target column
		if test is not None:
			self._test = test
		
			# check if target column user input else autodetect
			if target_column_name is None:
				self.target_column_name = self._auto_detect_target()
			else:
				self.target_column_name = target_column_name
		else:
			self._test = None
			if target_column_name is not None:
				self.target_column_name = target_column_name
			else:
				warnings.warn('No Target column defined')
				self.target_column_name = None
		
		# Get train size and validation size
		self._train_size = len(self._train)
		
		# Join data and separate target
		data, target = self._join_df()
		
		# Reduce memory usage of the data
		if reduce_memory:
			data = reduce_mem_usage(data)
			
		self._target = target
		OPFrame.__init__(self, data)
	
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
		if self.target_column_name is None:
			return self._train, None
		
		# Handling cases with test data
		if self._test is not None:
			column_order = self._test.columns
			common_columns = list(set(self._train.columns) & set(self._test.columns))
		
			data = pd.concat([self._train[common_columns], self._test[common_columns]], axis=0, sort=False)[column_order]
			target = self._train[self.target_column_name]
			
			return data, target
		else:
			# When no test data given
			return self._train.drop(self.target_column_name, axis=1), self._train[self.target_column_name]
	
	@property
	def train(self):
		"""
		Returns the training data
		
		:return: training data
		"""
		return self._data.iloc[:self._train_size]
	
	@property
	def target(self):
		"""
		Returns the target (Present in the training data

		:return: Target column in the training data
		"""
		return self._target
	
	@property
	def test(self):
		"""
		Returns the test data
		
		:return: test data
		"""
		if self._test is None:
			print(f'No Test data detected')
			return None
		return self._data.iloc[self._train_size:]
	
	def drop_cols(self, cols):
		"""
		Drop list of columns from the canvas df
		
		:param cols: List of columns to drop
		"""
		
		self._data.drop(columns=cols, inplace=True)
	
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
