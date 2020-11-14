from pandora.util.stages.estimation import fit, predict


def handle_train_estimator(estimator, features, target, **kwargs):
    return fit(estimator, features, target, **kwargs)


def handle_test_estimator(estimator, features):
    return predict(estimator, features)
