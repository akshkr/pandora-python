import pandas as pd


def make_balanced_df(df, target, factor):
	"""
	
	:param df:
	:param target:
	:param factor:
	:return:
	"""
	# Min and max value to make scale factor
	max_value = df[target].value_counts().max()
	min_value = df[target].value_counts().min()
	
	# Shuffling df and breaking
	df = df.sample(frac=1)
	
	# Calculate scale factor and scale
	scale_factor = int((max_value/min_value)*factor)
	print(f'Balancing with factor : {scale_factor}')
	replica = [df[df[target] == df[target].value_counts().argmin()]] * scale_factor
	
	balanced_df = pd.concat([df] + replica, axis=0)
	balanced_df = balanced_df.sample(frac=1)
	return balanced_df
