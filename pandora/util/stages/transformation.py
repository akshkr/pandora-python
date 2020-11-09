def fit_transform(operator, vector):
    # If the operator is function then map
    # Else fit_transform
    if callable(operator):
        return map(operator, vector), operator

    operator_obj = operator
    values = operator_obj.fit_transform(vector)
    return values, operator_obj


def transform(operator, vector):
    if callable(operator):
        return map(operator, vector)

    operator_obj = operator
    values = operator_obj.transform(vector)
    return values
