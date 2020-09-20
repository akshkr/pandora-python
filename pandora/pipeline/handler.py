from ..util.transformation import transform, fit_transform
from ..util.validate import kfold_validation


def handle_train_transformer(df, transformer, column):
	"""
		Handle transformations

		Args:
			df (pd.DataFrame): training DataFrame
			transformer (object): Object of transformer class
			column (str): Column name to transform

		Returns:
			Transformed values according to the transformer
		"""
	try:
		if isinstance(transformer, list):
			for i in transformer[:-1]:
				df[column], _ = fit_transform(df, i, column)
			transformer = transformer[-1]
		
		transformed_values, model = fit_transform(df, transformer, column)
		
		return [transformed_values, model]
	
	except Exception as ex:
		print(f'Exception encountered in {transformer} column {column}')
		raise ex


def handle_test_transformer(obj, df, pkey, transformer, column):
	"""
	Handle transformations
	
	Args:
		obj (object): Object to store model
		df (pd.DataFrame): training DataFrame
		pkey (str): Primary key to store model in obj
		transformer (object): Object of transformer class
		column (str): Column name to transform

	Returns:
		Transformed values according to the transformer
	"""
	try:
		# Handle case of transformer being list
		if isinstance(transformer, list):
			for i in transformer[:-1]:
				df[column], _ = transform(df, i, column)
			if callable(transformer[-1]):
				transformer_obj = transformer[-1]
			else:
				transformer_obj = obj.model[pkey]
			
		# If transformer is function
		elif callable(transformer):
			transformer_obj = transformer
			
		# If transformer is Class
		else:
			transformer_obj = obj.model[pkey]
		
		transformed_values, _ = transform(df, transformer_obj, column)
		
		return transformed_values
	
	except Exception as ex:
		print(f'Exception encountered in {transformer} column {column}')
		raise ex
		
		
def handle_train_estimator(obj, estimator, pkey, features, target, accuracy_func, k_fold=None):
	"""
	Handling estimator with validation for training
	
	Args:
		obj (object): Pipeline object
		estimator (object): Estimator object
		pkey (str): Primary key to store model
		features (np.ndarray): 2D numpy array of independent variables
		target (pd.Series): Dependent variable
		accuracy_func (function): function to check accuracy and validate
		k_fold (int): Number of fold validation
	"""
	if k_fold is not None:
		kfold_validation(
			k_fold, model_obj=estimator, model_args={}, features=features,
			target=target, accuracy_check=accuracy_func)
	
	# Return the model after training on the entire data
	model = estimator
	model.fit(features, target)
	obj.model[pkey] = model
	
	
def handle_test_estimator(obj, pkey, features):
	"""
	Handle estimator for test
	
	Args:
		obj (object): Pipeline object
		pkey (str): Primary key to store model
		features (np.ndarray): 2D numpy array of independent variables

	Returns:
		predictions
	"""
	estimator = obj.model[pkey]
	return estimator.predict(features)