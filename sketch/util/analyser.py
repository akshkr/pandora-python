from sketch.frame.mainframe import Canvas
from itertools import groupby
import pandas as pd
import seaborn as sns
import re


def _dist_analyser(data):
	"""
	
	:param data:
	:return:
	"""
	if len(pd.unique(data.values.ravel("K"))) > int(0.1 * len(data)):
		return 'continuous'
	
	else:
		return sorted(list(pd.unique(data.values.ravel("K"))))
	
	
def data_distribution(data_series):
	"""
	
	:param data_series:
	:return:
	"""
	n_unique = data_series.nunique()
	d_type = str(data_series.dtypes)
	
	# If the data is discrete return 1
	if n_unique < int(0.1 * len(data_series)):
		if n_unique < 50:
			return 1, list(data_series.unique())
		else:
			return 1, 'More than 50 records'
	
	# If data is continuous return 0
	elif d_type.startswith('int') or d_type.startswith('float'):
		if n_unique == len(data_series):
			return 0, 'Unique Record'
		else:
			return 0, {'min': data_series.min(), 'max': data_series.max()}
	else:
		return 0, 'objects'
	
	
def highlight_cells():
	# provide your criteria for highlighting the cells here
	return 'background-color: yellow'


def color_negative_red(val):
	"""
	Takes a scalar and returns a string with
	the css property `'color: red'` for negative
	strings, black otherwise.
	"""
	val = re.split('[^a-zA-Z]', val)[0]
	global color_dict
	
	# print(val)
	color = s[color_dict[val]]
	return f'background-color: {color}'
	# color = 'red'
	# return 'color: %s' % color


def analyse(obj):
	"""
	
	:param obj:
	:return:
	"""
	
	if isinstance(obj, Canvas):
		data = obj.data.round(4)
	elif isinstance(obj, pd.DataFrame):
		data = obj.round(4)
	else:
		raise TypeError(f'Type of {obj} : {type(obj)} is not supported. Pass a DataFrame or Canvas')

	column_list = list(data.columns)
	
	# Grouping column list according to their similar names
	res = [list(i) for j, i in groupby(column_list, lambda x: re.split('[^a-zA-Z]', x)[0])]
	res = [sorted(x) for x in res if len(x) > 1]
	global color_dict
	color_dict = dict()
	ind = 0
	for x in res:
		color_dict[str(re.split('[^a-zA-Z]', x[0])[0])] = ind
		ind += 1
	
	#
	result_df = pd.DataFrame({'Columns': column_list})
	result_df['Distribution'], result_df['Values'] = zip(*result_df['Columns'].map(lambda x: data_distribution(data[x])))
	# global color_dict
	stylize(result_df, res, color_dict)
	
	
def stylize(result_df, res, color_dict):
	
	temp_df = result_df.copy()
	result_df = result_df.style
	eval_str = f"result_df"
	i = 0
	for g in res:
		color = s[i]
		g_df = temp_df.loc[temp_df['Columns'].str.contains('|'.join(g))].index
		eval_str = f"{eval_str}.applymap(color_negative_red, subset=pd.IndexSlice[{g_df.min()}:{g_df.max()}, ['Columns']])"
		global idx
		idx += 1
	
	print(eval_str)
	eval(eval_str)
	
	print('applied while converting')
	result_df.to_excel('ejemplo.xlsx', engine='openpyxl', encoding='utf-8', index=False)


s = sns.color_palette("pastel", 10)
s = s.as_hex()
idx = 0
d = pd.read_csv('/Users/akashkumar/Workspace/Data Science/data/ieee-fraud-detection/train_transaction.csv', nrows=10000)
# d = pd.read_csv('/Users/akashkumar/Workspace/Data Science/data/av_India_ml/train.csv')
analyse(d)
