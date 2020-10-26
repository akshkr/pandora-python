from pandora.util.stages.transformation import fit_transform


def handle_train_preprocessor(preprocessor, feature):
    preprocessor_list = list()

    if isinstance(preprocessor, list):
        for i in preprocessor[:-1]:
            feature, operator = fit_transform(i, feature)
            preprocessor_list.append(operator)
        preprocessor = preprocessor[-1]

    transformed_values, models = fit_transform(preprocessor, feature)
    if preprocessor_list:
        models = [*preprocessor_list, models]

    return [transformed_values, models]
