from pandora.util.stages.transformation import fit_transform, transform


def handle_train_preprocessor(preprocessor, feature):
    # preprocessor_list = list()

    # If input preprocessors are a list of preprocessor
    # Run N-1 preprocessor and append the list of trained preprocessor
    if isinstance(preprocessor, list):
        for i in preprocessor[:-1]:
            feature = fit_transform(i, feature)

            # 201109: Uncomment this to get models
            # preprocessor_list.append(operator)
        preprocessor = preprocessor[-1]

    # If input is one preprocessor/ last preprocessor of the list
    # Run and append the list of models to the N-1 models
    transformed_values = fit_transform(preprocessor, feature)

    # 201109: Uncomment this to get models
    # if preprocessor_list:
    #     models = [*preprocessor_list, models]

    return [transformed_values]


def handle_test_preprocessor(preprocessor, feature):
    if isinstance(preprocessor, list):
        for i in preprocessor[:-1]:
            feature = transform(i, feature)
        preprocessor = preprocessor[-1]

    transformed_values = transform(preprocessor, feature)

    return [transformed_values]
