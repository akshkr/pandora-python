def fit_transform(df, transformer, column, params):
	
	if callable(transformer):
		return df[column].apply(lambda x: transformer(x)), None
	else:
		t = transformer
		values = t.fit_transform(df[column])
		return values, t
	

def transform(df, transformer, column, params):
	
	if callable(transformer):
		return df[column].apply(lambda x: transformer(x)), None
	else:
		t = transformer
		values = t.transform(df[column])
		return values, t
