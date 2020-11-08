def fit_transform(operator, vector):
	# If the operator is function then map
	# Else fit_transform
	if callable(operator):
		return map(operator, vector), operator

	else:
		op = operator
		values = op.fit_transform(vector)
		return values, op
	

def transform(operator, vector):
	if callable(operator):
		return map(operator, vector)

	else:
		op = operator
		values = op.transform(vector)
		return values
