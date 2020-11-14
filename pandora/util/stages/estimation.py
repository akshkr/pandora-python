def fit(estimator, features, target, **kwargs):
    model = estimator
    model.fit(features, target, **kwargs)

    return model


def predict(estimator, features):
    return estimator.predict(features)
