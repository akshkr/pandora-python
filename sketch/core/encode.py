"""
Functions to encode given list of columns using different methods
"""

import pandas as pd


def dummy_encoder(df, columns, drop):
	"""
	Encodes the given column using One Hot encoder
	"""
	encoded_data = pd.get_dummies(df[columns])
	
	if drop:
		df = df.drop(columns=columns)
		
	df = pd.concat([df, encoded_data], axis=1)
	return df


def label_encoder(df, columns, fill_with):
	"""
	Encodes the given list of columns using Label encoder.
	"""
	from sklearn.preprocessing import LabelEncoder
	encoder = LabelEncoder()
	
	# Handling Nan values
	df[columns].fillna(fill_with, inplace=True)
	df[columns] = df[columns].astype(str)
	
	# Encoding using Label Encoder
	for col in columns:
		df[col] = encoder.fit_transform(df[col])
	
	# Converting the data type to categorical
	df[columns] = df[columns].astype('category')
	
	return df


def frequency_encoder(df, columns, drop):
	"""
	Encodes the given list of columns using frequency
	of the value in the entire column.
	"""
	for column in columns:
		
		encoded_column = df[column].value_counts().to_dict()
		df[f'{column}_freq_enc'] = df[column].map(encoded_column)
		
	if drop:
		df = df.drop(columns=columns)
		
	return df
