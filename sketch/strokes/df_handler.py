import pandas as pd


def make_balanced_df(df, target, factor):
	"""
	
	:param df:
	:param target:
	:param factor:
	:return:
	"""
	max_value = df[target].value_counts().max()
	min_value = df[target].value_counts().min()
	
	scale_factor = int((max_value/min_value)*factor)
	print(f'Balancing with factor : {scale_factor}')
	replica = [df[df[target] == df[target].value_counts().argmin()]] * scale_factor
	
	new_df = pd.concat([df] + replica, axis=0)
	new_df = new_df.sample(frac=1)
	return new_df
