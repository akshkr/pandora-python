import inspect


def handle_transformer(df, transformer, column, params):
	"""
	
	Args:
		df:
		transformer:
		column:
		params:

	Returns:

	"""
	if callable(transformer):
		return transformer(df[column], **params)
	else:
		t = transformer
		return t.fit_transform(df[column])
	
	
def validate_transformer(transformers, inner_list=False):
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