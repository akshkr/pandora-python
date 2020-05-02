import numpy as np


def reduce_mem_usage(df):
	"""
	Reduces the size of the DataFrame by reducing the data type of the series
	
	Args:
		df (pd.DataFrame): DataFrame whose size is to be reduced

	Returns:
		DataFrame with reduced memory
	"""
	numerals = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
	start_mem = df.memory_usage().sum() / 1024**2
	for col in df.columns:
		col_type = df[col].dtypes
		if col_type in numerals:
			c_min = df[col].min()
			c_max = df[col].max()
			if str(col_type).startswith('int'):
				if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
					df[col] = df[col].astype(np.int8)
				elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
					df[col] = df[col].astype(np.int16)
				elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
					df[col] = df[col].astype(np.int32)
				elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
					df[col] = df[col].astype(np.int64)
			else:
				if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
					df[col] = df[col].astype(np.float16)
				elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
					df[col] = df[col].astype(np.float32)
				else:
					df[col] = df[col].astype(np.float64)
	
	end_mem = df.memory_usage().sum() / 1024**2
	print(
		f'Mem. usage decreased to {end_mem.round(2)} Mb'
		f' {(100 * (start_mem - end_mem) / start_mem).round(2)}% reduction)'
	)
		
	original_df = df.copy()
	for col in df.columns:
		if df[col].dtype != 'O':
			if (df[col] - original_df[col]).sum() != 0:
				df[col] = original_df[col]
				print(f'Bad transformation of {col}. Reverting...')
	return df
