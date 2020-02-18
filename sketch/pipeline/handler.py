from sketch.util.validate import kfold_validation


def handle_transformer(obj, df, pkey, transformer, column, params):
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
	if callable(transformer):
		return transformer(df[column], **params)
	else:
		t = transformer
		values = t.fit_transform(df[column])
		obj.model[pkey] = t
		return values
	
	
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
			if not (hasattr(t, "fit") or hasattr(t, "fit_transform")) or \
				not hasattr(t, "transform"):
				raise TypeError(
					f"All intermediate steps should be "
					f"transformers and implement fit and transform "
					f"{t} {type(t)} doesnt")
	
			
def handle_estimator(obj, estimator, pkey, features, target, params, accuracy_func):
	"""
	
	Args:
		obj (object): Pipeline object
		estimator (object): Estimator object
		pkey (str): Primary key to store model
		features (np.ndarray): 2D numpy array of independent variables
		target (pd.Series): Dependent variable
		params (dict): Parameters
		accuracy_func (function): function to check accuracy and validate

	Returns:
		model
	"""
	model = kfold_validation(
		4, model_class=estimator, model_args={}, features=features,
		target=target, accuracy_check=accuracy_func)
	
	obj.model[pkey] = model
	return model
