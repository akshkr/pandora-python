from pandas.util.testing import assert_frame_equal
from sketch.frame.operation import OPFrame
import pandas as pd
import string
import random


def test_operations():
	
	upper_case = string.ascii_uppercase
	classes = ['A', 'B', 'C']
	a = [random.choice(upper_case) for i in range(100)]
	b = [random.choice(classes) for i in range(100)]
	
	df = pd.DataFrame({
		'A': a,
		'B': b
	})
	
	opframe_obj = OPFrame(data=df)
	
	test_df = df.copy()
	test_df['B_A'] = 0
	test_df['B_B'] = 0
	test_df['B_C'] = 0
	test_df.loc[test_df['B'] == 'A', "B_A"] = 1
	test_df.loc[test_df['B'] == 'B', "B_B"] = 1
	test_df.loc[test_df['B'] == 'C', "B_C"] = 1
	test_df.drop('B', axis=1, inplace=True)
	cols = test_df.columns
	
	opframe_obj.one_hot_encode(['B'])
	output_df = opframe_obj.data
	
	assert list(test_df.columns.sort_values()) == list(output_df.columns.sort_values())
	assert_frame_equal(test_df, output_df[cols], check_dtype=False)
