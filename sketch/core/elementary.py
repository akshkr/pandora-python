"""
Function to modify values in a column of DataFrame
"""
import numpy as np


def count_remove(df, column, min_count=None, max_count=None, min_percentage=None, max_percentage=None):
	"""
	Remove values with the count according to the passed parameter
	"""
	retain_values = df[column].value_counts()
	total_rows = len(df)
	
	if min_count is not None:
		retain_values = retain_values[retain_values >= min_count]
	elif max_count is not None:
		retain_values = retain_values[retain_values <= max_count]
	elif min_percentage is not None:
		retain_values = retain_values[retain_values >= min_percentage * total_rows]
	elif max_percentage is not None:
		retain_values = retain_values[retain_values <= max_percentage * total_rows]

	retain_values = list(retain_values.index)
	return np.where(df[column].isin(retain_values), df[column], np.nan)
