from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_boston
from xgboost import XGBRegressor

from pandora.util.io import save_model, load_model
from pandora import CompositePipeline

X, y = load_boston(return_X_y=True)


def test_single_preprocessor():
    tp = CompositePipeline()
    tp.add(columns=range(13))
    tp.compile(
        estimator=XGBRegressor(random_state=3)
    )

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=3)
    tp.run(x_train, y_train)
    y_pred = tp.predict(x_test)

    assert mean_squared_error(y_test, y_pred) < 20
