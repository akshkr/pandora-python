def validate_transformer(transformers, transformer_list_flag=False):
	"""
	Validate list of transformers

	Args:
		transformers (list): List of transformers
		transformer_list_flag (bool): True is transformer is a list
	"""
	for t in transformers:
		if callable(t):
			pass
		
		elif isinstance(t, list) and not transformer_list_flag:
			validate_transformer(t, transformer_list_flag=True)
		
		else:
			# The transformer must implement fit or fit_transform and transform
			if not (hasattr(t, "fit") or hasattr(t, "fit_transform")) or \
				not hasattr(t, "transform"):
				raise TypeError(
					f"All intermediate steps should be "
					f"transformers and implement fit and transform "
					f"{t} {type(t)} doesnt")