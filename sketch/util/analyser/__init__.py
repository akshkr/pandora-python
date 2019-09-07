from sketch.util.analyser.distribution import data_distribution
from sketch.util.analyser.distribution import null_percentage
from sketch.frame.mainframe import Canvas
from itertools import groupby
import pandas as pd
import seaborn as sns
import re


s = sns.color_palette("pastel", 10)
s = s.as_hex()
idx = 0


def color_groups(val):
	"""
	Takes a value and assigns the background color according to
	the color dictionary initialized
	"""
	val = re.split('[^a-zA-Z]', val)[0]
	color = s[color_dict[val]]
	
	return f'background-color: {color}'


def analyse(obj, path='analyzed_df.xlsx'):
	"""
	Main analyser function to be called on a DataFrame or a Canvas object

	:param obj: DataFrame or a Canvas object to be analysed
	:param path:
	"""
	
	# Check for the type of object passed to analyse
	if isinstance(obj, Canvas):
		data = obj.data.round(4)
	elif isinstance(obj, pd.DataFrame):
		data = obj.round(4)
	else:
		raise TypeError(f'Type of {obj} : {type(obj)} is not supported. Pass a DataFrame or Canvas')
	
	# Getting the list of Columns in the DataFrame
	column_list = list(data.columns)
	
	# Grouping column list according to their similar names
	# Groups are split if their staring consecutive alphabets are same (Case Sensitive)
	res = [list(i) for j, i in groupby(column_list, lambda x: re.split('[^a-zA-Z]', x)[0])]
	res = [sorted(x) for x in res if len(x) > 1]
	
	# Color dictionary for Grouped columns. They are to be accessed by the color encoder function
	global color_dict
	color_dict = dict()
	ind = 0
	for x in res:
		color_dict[str(re.split('[^a-zA-Z]', x[0])[0])] = ind
		ind += 1
	
	# Making DataFrame with columns : Columns, Distribution and Values
	result_df = pd.DataFrame({'Columns': column_list})
	result_df['Distribution'], result_df['Values'], result_df['DType'] = zip(
		*result_df['Columns'].map(lambda x: data_distribution(data[x])))
	
	result_df = null_percentage(data, result_df)
	
	# Style the DataFrame and save in Excel format
	styled_df = stylize(result_df, res)
	styled_df.to_excel(path, engine='openpyxl', encoding='utf-8', index=False)


def stylize(df, res):
	"""
	Performs various types of color encoding and formatting on DataFrame and saves it into Excel file

	:param df: The input DataFrame to perform styling
	:param res: List of grouped columns
	"""
	
	indexer = df.copy()
	styled_df = df.style
	
	# The string to be evaluated for styling
	style_string = f"styled_df"
	
	# Color encoding the similar group of columns
	for g in res:
		g_df = indexer.loc[indexer['Columns'].str.contains('|'.join(g))].index
		style_string = f"{style_string}.applymap(color_groups, subset=pd.IndexSlice[{g_df.min()}:{g_df.max()}, ['Columns']])"
	
	# Evaluate the style string
	eval(style_string)
	return styled_df


d = pd.read_csv('/Users/akashkumar/Workspace/Data Science/data/ieee-fraud-detection/train_transaction.csv', nrows=10000)
# d = pd.read_csv('/Users/akashkumar/Workspace/Data Science/data/av_India_ml/train.csv')
analyse(d)
