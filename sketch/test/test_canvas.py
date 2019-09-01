from pandas.util.testing import assert_frame_equal, assert_series_equal
from sketch.frame.mainframe import Canvas
import pandas as pd
import numpy as np
import string
import random


def test_all_args():
	
	a = np.random.randint(low=1, high=50, size=500)
	b = np.random.randint(low=1, high=50, size=500)
	
	lower_case = string.ascii_lowercase
	t = [random.choice(lower_case) for i in range(500)]
	
	total = pd.DataFrame({
		'A': a,
		'B': b,
		'T': t
	})
	
	train = total.iloc[: int(len(total)*0.75)]
	test = total.iloc[int(len(total)*0.75):].drop('T', axis=1)
	canvas_obj = Canvas(train, test)
	
	# test Canvas data
	test_df = total.drop('T', axis=1)
	output_df = canvas_obj.data.sort_index()
	assert_frame_equal(test_df, output_df, check_exact=True, check_dtype=False)
	
	# Test Canvas Train
	test_df = train.drop('T', axis=1)
	output_df = canvas_obj.train.sort_index()
	assert_frame_equal(test_df, output_df, check_exact=True, check_dtype=False)

	# Test Canvas Target
	test_df = train['T']
	output_df = canvas_obj.target.sort_index()
	assert_series_equal(test_df, output_df, check_exact=True, check_dtype=False)

	# Test Canvas test
	test_df = test
	output_df = canvas_obj.test.sort_index()
	assert_frame_equal(test_df, output_df, check_exact=True, check_dtype=False)


def test_one_data():
	
	a = np.random.randint(low=1, high=50, size=500)
	b = np.random.randint(low=1, high=50, size=500)
	
	lower_case = string.ascii_lowercase
	t = [random.choice(lower_case) for i in range(500)]
	
	train = pd.DataFrame({
		'A': a,
		'B': b,
		'T': t
	})
	
	canvas_obj = Canvas(train, None, target_column_name='T')
	
	# test Canvas data
	test_df = train.drop('T', axis=1)
	output_df = canvas_obj.data.sort_index()
	assert_frame_equal(test_df, output_df, check_exact=True, check_dtype=False)
	
	# Test Canvas Train
	test_df = train.drop('T', axis=1)
	output_df = canvas_obj.train.sort_index()
	assert_frame_equal(test_df, output_df, check_exact=True, check_dtype=False)
	
	# Test Canvas Target
	test_df = train['T']
	output_df = canvas_obj.target.sort_index()
	assert_series_equal(test_df, output_df, check_exact=True, check_dtype=False)
	
	# Test Canvas test
	output_df = canvas_obj.test
	assert output_df is None
