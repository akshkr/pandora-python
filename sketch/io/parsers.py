import pandas as pd


def outline(df1, df2):
	if isinstance(df1, pd.DataFrame) and isinstance(df2, pd.DataFrame):
		df1, df2 = df1, df2
	else:
		raise TypeError(f'Type of {df1} : {type(df1)} is not supported. Pass a DataFrame')
	
	from sketch.frame.frames import Canvas
	return Canvas(df1, df2)