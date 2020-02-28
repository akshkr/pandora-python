from ..util.validate import kfold_validation
from ..util.transformation import transform, fit_transform


def handle_train_transformer(obj, df, pkey, transformer, column, params):
	"""
		Handle transformations

		Args:
			obj (object): Object to store model
			df (pd.DataFrame): training DataFrame
			pkey (str): Primary key to store model in obj
			transformer (object): Object of transformer class
			column (str): Column name to transform
			params (dict): parameters to pass in a function

		Returns:
			Transformed values according to the transformer
		"""
	try:
		if isinstance(transformer, list):
			for i in transformer[:-1]:
				df[column], _ = fit_transform(df, i, column, params)
			transformer = transformer[-1]
		
		prediction_values, model = fit_transform(df, transformer, column, params)
		if model is not None:
			obj.model[pkey] = model
			
		return prediction_values
	
	except Exception as ex:
		print(f'Exception encountered in {transformer} column {column}')
		raise ex


def handle_test_transformer(obj, df, pkey, transformer, column, params):
	"""
	Handle transformations
	
	Args:
		obj (object): Object to store model
		df (pd.DataFrame): training DataFrame
		pkey (str): Primary key to store model in obj
		transformer (object): Object of transformer class
		column (str): Column name to transform
		params (dict): parameters to pass in a function

	Returns:
		Transformed values according to the transformer
	"""
	try:
		# Handle case of transformer being list
		if isinstance(transformer, list):
			for i in transformer[:-1]:
				df[column], _ = transform(df, i, column, params)
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
		prediction_values, _ = transform(df, transformer_obj, column, params)

	except Exception as ex:
		print(f'Exception encountered in {transformer} column {column}')
		raise ex
		
	
def validate_transformer(transformers, transformer_list=False):
	"""
	Validate list of transformers
	
	Args:
		transformers (list): List of transformers
		transformer_list (bool): True is transformer is a list
	"""
	for t in transformers:
		if callable(t):
			pass
		
		elif isinstance(t, list) and not transformer_list:
			validate_transformer(t, transformer_list=True)
			
		else:
			# The transformer must implement fit or fit_transform and transform
			if not (hasattr(t, "fit") or hasattr(t, "fit_transform")) or \
				not hasattr(t, "transform"):
				raise TypeError(
					f"All intermediate steps should be "
					f"transformers and implement fit and transform "
					f"{t} {type(t)} doesnt")
	
			
def handle_estimator(obj, estimator, pkey, features, target, params, accuracy_func, test=False, k_fold=None):
	"""
	Handling estimator with validation
	
	Args:
		obj (object): Pipeline object
		estimator (object): Estimator object
		pkey (str): Primary key to store model
		features (np.ndarray): 2D numpy array of independent variables
		target (pd.Series): Dependent variable
		params (dict): Parameters
		accuracy_func (function): function to check accuracy and validate
		test (bool): True is estimating for test set
		k_fold (int): Number of fold validation

	Returns:
		model
	"""
	
	if test:
		estimator = obj.model[pkey]
		return estimator.predict(features)
	
	else:
		if k_fold is not None:
			kfold_validation(
				k_fold, model_obj=estimator, model_args={}, features=features,
				target=target, accuracy_check=accuracy_func)
		
		# Return the model after training on the entire data
		model = estimator
		model.fit(features, target)
		obj.model[pkey] = model
		return model
