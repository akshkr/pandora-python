def min_max_scale(df, col):
	min_c = df[col].min()
	max_c = df[col].max()
	
	df[col] = (df[col] - min_c) / (max_c - min_c)
	
	return df[col]
