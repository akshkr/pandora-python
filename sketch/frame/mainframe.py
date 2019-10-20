from sketch.util.dfhandler import reduce_mem_usage
from sketch.frame.operation import OPFrame
import pandas as pd
import warnings


class Canvas(OPFrame):
	"""
	Frame with functionality to preprocess large train and test data clubbed
	together. This frame is compatible with various models and analysing tools
	added in Sketch.
	
	Parameters
	----------
	
	train : DataFrame
		Training DataFrame which contains target
	test : DataFrame
		Test DataFrame which doesn't contains target of for which we need
		predictions
	target_column_name : String
		Name of the column which acts as the Target/ Dependent Variable.
		If not passed, will be auto detected
	reduce_memory : boolean, default False
		Whether to reduce the size of the complete data by reducing the
		data types
	"""
	
	def __init__(
		self,
		train,
		test,
		target_column_name,
		reduce_memory=False
	):
		
		# Shuffle and initialize train
		self._train = train.sample(frac=1).reset_index(drop=True)
		
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
		print(f'Auto Detecting Target...')
		target_column = list(set(self._train.columns) - set(self._test.columns))
		
		if len(target_column) > 1:
			raise ValueError(f'Found more than one target : {target_column}')
		else:
			print(f'Target Column detected : {target_column[0]}')
			return target_column[0]
	
	def _join_df(self):
		"""
		Joins the train and test and fill test target as 0
		"""
		# If no target column
		if self.target_column_name is None:
			return self._train, None
		
		# Handling cases with test data
		if self._test is not None:
			# TODO: Handle for all data types
			self._test[self.target_column_name] = 0
			
			data = pd.concat([self._train, self._test], axis=0)
			target = self._train[self.target_column_name]
			
			return data, target
		else:
			# When no test data given
			return self._train, self._train[self.target_column_name]
	
	@property
	def train(self):
		"""
		Returns the training data
		
		Returns
		-------
		Return the training data without the target column
		"""
		return self._data.iloc[:self._train_size].drop(columns=[self.target_column_name])
	
	@property
	def target(self):
		"""
		Returns the target (Present in the training data)
		
		Returns
		-------
		Returns the target column of the training data
		"""
		return self._data.iloc[:self._train_size][self.target_column_name]
	
	@property
	def test(self):
		"""
		Returns the test data
		
		Returns
		-------
		Returns the test data if present
		"""
		if self._test is None:
			print(f'No Test data detected')
			return None
		return self._data.iloc[self._train_size:].drop(columns=[self.target_column_name])
	
	def drop_cols(self, cols):
		"""
		Drops columns from the data
		
		Parameters
		----------
		
		cols : list of columns
			List of columns to drop from the data
		"""
		
		self._data.drop(columns=cols, inplace=True)
