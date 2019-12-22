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


def color_null(val):
	"""

	:param val:
	"""
	null_palette = sns.diverging_palette(140, 0, s=99, l=40, n=10).as_hex()
	color = null_palette[int(val * 10)]
	
	return f'color: {color}'


def _stylize(df, res):
	"""
	Performs various types of color encoding and formatting on DataFrame and saves it into Excel file

	:param df: The input DataFrame to perform styling
	:param res: List of grouped columns
	"""
	
	indexer = df.copy()
	
	# Do not remove this. Gets evaluated as string
	styled_df = df.style
	
	# The string to be evaluated for styling
	style_string = f"styled_df"
	
	# Color encoding the similar group of columns
	for g in res:
		g_df = indexer.loc[indexer['Columns'].str.contains('|'.join(g))].index
		style_string = f"{style_string}.applymap(color_groups, subset=pd.IndexSlice[{g_df.min()}:{g_df.max()}, ['Columns']])"
	
	style_string = f'{style_string}.applymap(color_null, subset=["Null_Percentage"])'
	
	# Evaluate the style string
	styled_df = eval(style_string)
	
	return styled_df


def analyse(obj, save=False, path='analyzed_df.xlsx'):
	"""
	Main analyser function to be called on a DataFrame or a Canvas object

	Parameters
	----------
	obj : Canvas or DataFrame
		Object to be analyzed
	save : boolean
		Save if True
	path : path
		path of analyzed excel file
		
	Returns
	-------
	styled_df : Styled DataFrame
		Styled DataFrame with all the analyzed information
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
	
	print(f'Number of records: {data.shape[0]}')
	print(data.head())
	
	# Grouping column list according to their similar names
	# Groups are split if their starting consecutive alphabets are same (Case Sensitive)
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
	styled_df = _stylize(result_df, res)
	
	if save:
		styled_df.to_excel(path, engine='openpyxl', index=False)
	
	return styled_df
