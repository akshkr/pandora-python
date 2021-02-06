from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_boston
from xgboost import XGBRegressor

from pandora.core.model.builder.non_parametric import NonParametricModelBuilder
from pandora.util import seed_everything
from pandora.pipeline import TabularPipeline

import logging
LOGGER = logging.getLogger(__name__)
seed_everything()

X, y = load_boston(return_X_y=True)


def test_all_columns():
    tp = TabularPipeline()
    tp.add(column=range(13))
    tp.compile(
        estimator=XGBRegressor(random_state=3)
    )

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=3)
    tp.run(x_train, y_train)
    y_pred = tp.predict(x_test)
    accuracy = mean_squared_error(y_test, y_pred)
    del tp

    LOGGER.info(f'Accuracy: {accuracy}')
    assert accuracy < 15


def test_single_preprocessor():
    tp = TabularPipeline()
    tp.add(MinMaxScaler(), column=[0])
    tp.add(column=range(1, 13))
    tp.compile(
        estimator=XGBRegressor(random_state=3)
    )

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=3)
    tp.run(x_train, y_train)
    y_pred = tp.predict(x_test)
    accuracy = mean_squared_error(y_test, y_pred)
    del tp

    LOGGER.info(f'Accuracy: {accuracy}')
    assert accuracy < 15


def test_multiple_preprocessor_different_column():
    tp = TabularPipeline()
    tp.add(MinMaxScaler(), column=[0, 2, 4])
    tp.add(StandardScaler(), column=[1, 3, 5])
    tp.add(column=range(6, 13))
    tp.compile(
        estimator=XGBRegressor(random_state=3)
    )

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=3)
    tp.run(x_train, y_train)
    y_pred = tp.predict(x_test)
    accuracy = mean_squared_error(y_test, y_pred)
    del tp

    LOGGER.info(f'Accuracy: {accuracy}')
    assert accuracy < 15


def test_functions():
    tp = TabularPipeline()
    tp.add(lambda x: x*10, column=range(9))
    tp.add(column=range(9, 13))

    tp.compile(
        estimator=XGBRegressor(random_state=3)
    )

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=3)
    tp.run(x_train, y_train)
    y_pred = tp.predict(x_test)
    accuracy = mean_squared_error(y_test, y_pred)
    del tp

    LOGGER.info(f'Accuracy: {accuracy}')
    assert accuracy < 15


def test_multiple_preprocessor_same_column():
    tp = TabularPipeline()
    tp.add([lambda x: x * 10, MinMaxScaler()], column=range(4))
    tp.add([MinMaxScaler(), StandardScaler()], column=range(4, 9))
    tp.add(column=range(9, 13))

    tp.compile(
        estimator=XGBRegressor(random_state=3)
    )

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=3)
    tp.run(x_train, y_train)
    y_pred = tp.predict(x_test)
    accuracy = mean_squared_error(y_test, y_pred)
    del tp

    LOGGER.info(f'Accuracy: {accuracy}')
    assert accuracy < 15


def test_model_builder():
    tp = TabularPipeline()
    tp.add(column=range(0, 13))

    tp.compile(
        estimator=NonParametricModelBuilder('xgbr', search='random')
    )

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=3)
    tp.run(x_train, y_train)
    y_pred = tp.predict(x_test)
    accuracy = mean_squared_error(y_test, y_pred)
    del tp

    LOGGER.info(f'Accuracy: {accuracy}')
    assert accuracy < 15


def test_grid_model_builder():
    tp = TabularPipeline()
    tp.add(column=range(0, 13))

    params = {
        "max_depth": [3, 8, 12],
        "min_child_weight": [3, 5, 7],
        "colsample_bytree": [0.5, 0.7],
    }

    tp.compile(
        estimator=NonParametricModelBuilder('xgbr', 'grid', params=params)
    )

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=3)
    tp.run(x_train, y_train)
    y_pred = tp.predict(x_test)
    accuracy = mean_squared_error(y_test, y_pred)
    del tp

    LOGGER.info(f'Accuracy: {accuracy}')
    assert accuracy < 15


def test_cv():
    tp = TabularPipeline()
    tp.add(column=range(0, 13))

    tp.compile(
        estimator=XGBRegressor(random_state=3)
    )

    tp.enable_cv('KFold', metrics=['mse'], n_split=4)

    tp.run(X, y)
