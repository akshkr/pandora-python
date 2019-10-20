import pandas as pd


def outline(df1, df2=None, target_column_name=None, reduce_memory=False):
	"""
	Makes a canvas object instance
	
	:param df1: DataFrame one, supposed to be training data
	:param df2: DataFrame two, supposed to be test data
	:param target_column_name: Name of the target column in the train data
	:param reduce_memory: Reduce the datatype for less memory consumption
	:return: Canvas object instance
	"""
	
	if not isinstance(df1, pd.DataFrame):
		raise TypeError(f'Type of {df1} : {type(df1)} is not supported. Pass a DataFrame')
	
	if not isinstance(df2, pd.DataFrame) and df2 is not None:
		raise TypeError(f'Type of {df2} : {type(df2)} is not supported. Pass a DataFrame')

	from sketch.frame.mainframe import Canvas
	return Canvas(df1, df2, target_column_name=target_column_name, reduce_memory=reduce_memory)
