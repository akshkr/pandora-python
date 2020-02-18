def fit_transform(df, transformer, column, params):
	
	if callable(transformer):
		return transformer(df[column], **params), None
	else:
		t = transformer
		values = t.fit_transform(df[column])
		return values, t
	

def transform(df, transformer, column, params):
	
	if callable(transformer):
		return transformer(df[column], **params), None
	else:
		t = transformer
		values = t.transform(df[column])
		return values, t
