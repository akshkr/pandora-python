from pandas.util.testing import assert_frame_equal
import pandas as pd
import os

from sketch.frame.mainframe import Canvas

data_folder = '/Users/akashkumar/Workspace/Data Science/Libs/data_preprocessor/sketch/test/data'
train_file_path = 'train.csv'
test_file_path = 'test.csv'
total_data_path = 'iris.csv'

train = pd.read_csv(os.path.join(data_folder, train_file_path))
test = pd.read_csv(os.path.join(data_folder, test_file_path))
total = pd.read_csv(os.path.join(data_folder, total_data_path))


def test_canvas():
	
	canvas_obj = Canvas(train, test)
	code_df = total.drop('Species', axis=1)
	cols = code_df.columns
	processed_df = canvas_obj.dataframe[cols].sort_values(by='Id').reset_index(drop=True)
	assert_frame_equal(code_df, processed_df, check_index_type=False)
