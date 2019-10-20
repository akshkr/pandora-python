import pandas as pd


def group_target_aggregate(can, columns):
	"""
	Groups and aggregate mean on columns
	
	:param can: canvas object
	:param columns: columns to aggregate
	:return: canvas object
	"""
	train_df = pd.concat([can.train, pd.Series(can.target)], axis=1)
	test_df = can.test
	
	for col in columns:
		map_dict = train_df.groupby([col])[can.target_column_name].agg(['mean']).reset_index().rename(
			columns={'mean': f'{col}_target_mean'}
		)
		
		map_dict.index = map_dict[col].values
		map_dict = map_dict[col+'_target_mean'].to_dict()
		
		train_df[f'{col}_target_mean'] = train_df[col].map(map_dict)
		test_df[f'{col}_target_mean'] = test_df[col].map(map_dict)
		
	data = pd.concat([train_df, test_df], axis=0)
	can._data = data
	
	return can
