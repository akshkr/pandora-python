from pandora.util.process import parallelize


def fit(estimator, features, target, **kwargs):
    model = estimator
    model.fit(features, target, **kwargs)

    return model


def predict(estimator, features):
    return estimator.predict(features)


def fit_all(estimators, features, target, n_jobs=None, bootstrap=False, index=None):
    if bootstrap:
        raise NotImplementedError(f'Bootstrap is not implemented. Issue raised at - '
                                  f'https://github.com/akshkr/pandora-python/issues/23')

    elif index:
        assert len(index) == len(estimators)

        features = [features[i] for i in index]
        target = [target[i] for i in index]

    else:
        if isinstance(features, list) and isinstance(target, list):
            assert len(features) == len(estimators) == len(target)

        else:
            raise TypeError('Invalid features and target')

    estimators = parallelize(fit, zip(estimators, features, target), n_jobs=n_jobs)

    return estimators
