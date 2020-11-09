from pandora.util.stages.estimation import fit, predict


def handle_train_estimator(estimator, features, target):
    return fit(estimator, features, target)


def handle_test_estimator(estimator, features):
    return predict(estimator, features)
