def fit(estimator, features, target):
    model = estimator
    model.fit(features, target)

    return model


def predict(estimator, features):
    return estimator.predict(features)
