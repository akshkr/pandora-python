from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from xgboost import XGBRegressor

from pandora import CompositePipeline

X, y = load_boston(return_X_y=True)


def test_all_columns():
    tp = CompositePipeline()
    tp.add(column=range(13))
    tp.compile(
        estimator=XGBRegressor(random_state=3)
    )

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=3)
    tp.run(x_train, y_train)
    y_pred = tp.predict(x_test)

    assert mean_squared_error(y_test, y_pred) < 20


def test_single_preprocessor():
    tp = CompositePipeline()
    tp.add(MinMaxScaler(), column=[0])
    tp.add(column=range(1, 13))
    tp.compile(
        estimator=XGBRegressor(random_state=3)
    )

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=3)
    tp.run(x_train, y_train)
    y_pred = tp.predict(x_test)

    assert mean_squared_error(y_test, y_pred) < 20


def test_multiple_preprocessor_different_column():
    tp = CompositePipeline()
    tp.add(MinMaxScaler(), column=[0, 2, 4])
    tp.add(StandardScaler(), column=[1, 3, 5])
    tp.add(column=range(6, 13))
    tp.compile(
        estimator=XGBRegressor(random_state=3)
    )

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=3)
    tp.run(x_train, y_train)
    y_pred = tp.predict(x_test)

    assert mean_squared_error(y_test, y_pred) < 20


def test_functions():
    tp = CompositePipeline()
    tp.add(lambda x: x*10, column=range(9))
    tp.add(column=range(9, 13))

    tp.compile(
        estimator=XGBRegressor(random_state=3)
    )

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=3)
    tp.run(x_train, y_train)
    y_pred = tp.predict(x_test)

    assert mean_squared_error(y_test, y_pred) < 20


def test_multiple_preprocessor_same_column():
    tp = CompositePipeline()
    tp.add([lambda x: x * 10, MinMaxScaler()], column=range(4))
    tp.add([MinMaxScaler(), StandardScaler()], column=range(4, 9))
    tp.add(column=range(9, 13))

    tp.compile(
        estimator=XGBRegressor(random_state=3)
    )

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=3)
    tp.run(x_train, y_train)
    y_pred = tp.predict(x_test)

    assert mean_squared_error(y_test, y_pred) < 20
