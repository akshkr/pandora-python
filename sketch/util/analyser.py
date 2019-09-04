from sketch.frame.mainframe import Canvas
from itertools import groupby
import pandas as pd
import re


def analyse(obj):
	if isinstance(obj, Canvas):
		data = obj.data
	elif isinstance(obj, pd.DataFrame):
		data = obj
	else:
		raise TypeError(f'Type of {obj} : {type(obj)} is not supported. Pass a DataFrame or Canvas')

	column_list = list(data.columns)
	
	res = [list(i) for j, i in groupby(column_list, lambda x: re.split('[^a-zA-Z]', x)[0])]
	res = [sorted(x) for x in res if len(x) > 1]
	
	s = [
		f'Columns "{x[0]} - {x[-1]}" ({len(x)} columns) : '
		f'has {sorted(pd.unique(data[x].values.ravel("K")))} values' for x in res
	]
	
	return s
