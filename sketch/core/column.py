"""
Performs operation relative to one column
"""


def group_column_operate(agg_df, data, aggregate_column, column, operation):
	"""
	Performs operation after grouping on a column
	
	Parameters
	----------
	agg_df : DataFrame
		DataFrame from which the mapping dictionary is evaluated
	data : DataFrame
		DataFrame in which mapping is done
	aggregate_column : string
		Column name of which aggregation is to be done
	column : string
		Column which is grouped
	operation : string (operation name)
		Operation name like mean, mode, etc.
	"""
	new_column_name = f'{column}_target_{operation}'
	map_dict = agg_df.groupby([column])[aggregate_column].agg([operation]).reset_index().rename(
		columns={operation: new_column_name}
	)
	
	map_dict.index = map_dict[column].values
	map_dict = map_dict[new_column_name].to_dict()
	
	# Map and create the new column
	data[new_column_name] = data[column].map(map_dict)

	return data
