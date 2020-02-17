def validate_column_names(df_columns, columns):
	
	if not set(columns).issubset(set(df_columns)):
		raise ValueError(
			f"These column don't exist in the DataFrame"
			f"{set(columns) - set(df_columns)}")
