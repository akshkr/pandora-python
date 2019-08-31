import pandas as pd


def outline(df1, df2=None):
	if not isinstance(df1, pd.DataFrame):
		raise TypeError(f'Type of {df1} : {type(df1)} is not supported. Pass a DataFrame')
	
	if not isinstance(df2, pd.DataFrame) or df2 is not None:
		raise TypeError(f'Type of {df2} : {type(df2)} is not supported. Pass a DataFrame')

	from sketch.frame.mainframe import Canvas
	return Canvas(df1, df2)
