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
		target_column_name=None,
		reduce_memory=False
	):
		
		# Shuffle training dataset and reset index
		# Just to make sure there is no bias in the positioning of data
		self._train = train.sample(frac=1).reset_index(drop=True)
		
		# Initialize test and target column
		
		# If test data is present and target column is not defined,
		# we try to auto detect target, if defined initialize directly
		# There is no option for test data input with no target column
		
		# If test data is not present, initialize target from user input
		# If no user input, warn that no target column is defined
		if test is not None:
			self._test = test.sample(frac=1).reset_index(drop=True)
		
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
		
		# NOTE: Make sure to reinitialize this variable if rows in training data decreases
		# Get train size and validation size
		self._train_size = len(self._train)
		
		# Join train and test data
		# The joined data contains target
		# target series only contains values of train data
		data, target = self._join_df()
		
		# Reduce memory usage of the data
		if reduce_memory:
			data = reduce_mem_usage(data)
			
		OPFrame.__init__(self, data)
		
		print(f'Train Data shape : {self.train.shape}')
		print(f'Test Data shape : {self.test.shape}')
	
	def _auto_detect_target(self):
		print(f'Auto Detecting Target...')
		target_column = list(set(self._train.columns) - set(self._test.columns))
		
		if len(target_column) > 1:
			raise ValueError(f'Found more than one target : {target_column}. Please pass target_column_name')
		else:
			print(f'Target Column detected : {target_column[0]}')
			return target_column[0]
	
	def _join_df(self):
		"""
		Joins the train and test and fill test target as 0
		
		Returns
		-------
		data : DataFrame
			train and test joined DataFrame along with target column
		target : Series
			target series only for train data
		"""
		# If no target column, return None as target
		if self.target_column_name is None:
			return self._train, None
		
		# Handling cases with test data
		# Here we fill target column in test data with zeros and concatenate
		# However note that the target series returned is of train data only
		if self._test is not None:
			# TODO: Handle for all data types
			self._test[self.target_column_name] = 0
			
			data = pd.concat([self._train, self._test], axis=0)
			
			return data, self._train[self.target_column_name]
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
		if self.target_column_name is not None:
			return self._data.iloc[:self._train_size][self.target_column_name]
		else:
			raise NameError(f'Target column is not defined while initialising Canvas object')
	
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
	
	def drop_cols(self, columns):
		"""
		Drops columns from the data
		
		Parameters
		----------
		columns : list of columns
			List of columns to drop from the data
		"""
		self._data.drop(columns=columns, inplace=True)
		
	def merge_data(self, column_name, train_df, test_df=None):
		"""
		Merge additional DataFrame to the Canvas object
		
		Parameters
		----------
		column_name : string
			Column name on which merge has to be done
		train_df : DataFrame
			Additional DataFrame of training data
		test_df : DataFrame
			Additional DataFrame of test data
		"""
		if test_df is not None:
			merge_df = pd.concat([train_df, test_df], axis=0)
		else:
			merge_df = train_df
		
		self._data = self._data.merge(merge_df, on=[column_name], how='left')
