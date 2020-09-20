def validate_column_names(df, columns):
	"""
	Validate columns in a DataFrame
	
	Args:
		df (pd.DataFrame): DataFrame to check column presence
		columns (list): List of columns to be checked
	"""
	
	if not set(columns).issubset(set(df.columns)):
		raise ValueError(
			f"These column don't exist in the DataFrame"
			f"{set(columns) - set(df.columns)}")
