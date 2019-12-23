def data_distribution(data_series):
	"""
	Takes pandas Series as input and returns Boolean (Discrete 1 Continuous 0)
	and Unique Values if discrete and

	:param data_series:
	:return:
	"""
	n_unique = data_series.nunique()
	d_type = str(data_series.dtypes)
	
	# If the data is discrete return 1
	if n_unique < int(0.1 * len(data_series)):
		
		dist = 'discrete'
		if n_unique < 50:
			values = list(data_series.unique())
		elif d_type.startswith('int') or d_type.startswith('float'):
			values = {'min': data_series.min(), 'max': data_series.max()}
		else:
			values = 'more than 50 records'
	
	# If data is continuous return 0
	elif d_type.startswith('int') or d_type.startswith('float'):
		
		dist = 'continuous'
		if n_unique == len(data_series):
			values = 'unique Record'
		else:
			values = {'min': data_series.min(), 'max': data_series.max()}
	else:
		dist = 'objects'
		values = 'objects'
	
	return dist, values, d_type


def null_percentage(df, result_df):
	"""
	
	:param df:
	:param result_df:
	:return:
	"""
	
	result_df['Null_Percentage'] = result_df['Columns'].apply(lambda x: (1 - df[x].count()/len(df)).round(3))
	return result_df


def repeat_percentage(df, result_df):
	"""
	
	"""
	result_df['Repeat_percentage'] = result_df['Columns'].apply(lambda x: df[x].nunique()).round(3)
	return result_df