def fit_transform(operator, vector):
	if callable(operator):
		return map(operator, vector), operator

	else:
		t = operator
		values = t.fit_transform(vector)
		return values, t
	

def transform(operator, vector):
	if callable(operator):
		return map(operator, vector)

	else:
		t = operator
		values = t.transform(vector)
		return values
