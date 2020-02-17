from sketch.util.validate import kfold_validation
import inspect


def handle_transformer(obj, df, pkey, transformer, column, params):
	"""
	
	Args:
		obj:
		df:
		pkey:
		transformer:
		column:
		params:

	Returns:

	"""
	if callable(transformer):
		return transformer(df[column], **params)
	else:
		t = transformer
		values = t.fit_transform(df[column])
		obj.model[pkey] = t
		return values
	
	
def validate_transformer(transformers, inner_list=False):
	"""
	
	Args:
		transformers:
		inner_list:

	Returns:

	"""
	for t in transformers:
		if callable(t):
			pass
		
		elif isinstance(t, list) and not inner_list:
			validate_transformer(t, inner_list=True)
			
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
		obj:
		estimator:
		pkey:
		features:
		target:
		params:
		accuracy_func:

	Returns:

	"""
	# n_splits, model_class, model_args, train_df, target, accuracy_check
	model = kfold_validation(4, model_class=estimator, model_args={}, features=features, target=target, accuracy_check=accuracy_func)
	obj[pkey] = model
	return model
