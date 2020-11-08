def fit(estimator, x, y):
    model = estimator
    model.fit(x, y)

    return model


def predict(estimator, x):
    return estimator.predict(x)
