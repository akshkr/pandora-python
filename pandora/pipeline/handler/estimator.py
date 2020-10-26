from pandora.util.stages.estimation import fit


def handle_train_estimator(estimator, features, target):
    return fit(estimator, features, target)
