from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from pandora import TextPipeline
import pytest
import re


@pytest.fixture
def newsgroup_data(number_of_categories):
    categories_list = [
        'comp.graphics',
        'alt.atheism',
        'rec.autos',
        'sci.crypt',
        'talk.politics.misc',
        'soc.religion.christian',
        'comp.os.ms - windows.misc',
        'rec.sport.baseball',
        'comp.sys.ibm.pc.hardware',
        'sci.med',
        'talk.politics.guns',
        'sci.space',
        'rec.sport.hockey',
        'talk.politics.mideast',
        'comp.sys.mac.hardware',
        'sci.electronics',
        'talk.religion.misc',
        'comp.windows.x',
        'rec.motorcycles',
        'misc.forsale'
    ]
    return fetch_20newsgroups(return_X_y=True, categories=categories_list[:number_of_categories])


@pytest.mark.parametrize("number_of_categories", [4])
def test_single_preprocessor(newsgroup_data):
    tp = TextPipeline()
    tp.add(TfidfVectorizer(max_features=1000))

    tp.compile(
        estimator=XGBClassifier(
            objective='multi:softmax', n_estimators=100, num_class=4, learning_rate=0.075,
            colsample_bytree=0.7, subsample=0.8, eval_metric='merror'
        )
    )
    x_train, x_test, y_train, y_test = train_test_split(newsgroup_data[0], newsgroup_data[1], test_size=0.33)
    tp.run(x_train, y_train)
    y_pred = tp.predict(x_test)

    assert accuracy_score(y_test, y_pred) > 0.9


@pytest.mark.parametrize("number_of_categories", [4])
def test_multi_preprocessor(newsgroup_data):
    tp = TextPipeline()
    tp.add(TfidfVectorizer(max_features=1000))
    tp.add(TfidfVectorizer(max_features=20))

    tp.compile(
        estimator=XGBClassifier(
            objective='multi:softmax', n_estimators=100, num_class=4, learning_rate=0.075,
            colsample_bytree=0.7, subsample=0.8, eval_metric='merror'
        )
    )
    x_train, x_test, y_train, y_test = train_test_split(newsgroup_data[0], newsgroup_data[1], test_size=0.33)
    tp.run(x_train, y_train)
    y_pred = tp.predict(x_test)

    assert accuracy_score(y_test, y_pred) > 0.9


@pytest.mark.parametrize("number_of_categories", [4])
def test_multi_preprocessor_with_func(newsgroup_data):
    def remove_symbols(text):
        return re.sub(r'^[a-zA-Z0-9]', '', text)

    tp = TextPipeline()
    tp.add(TfidfVectorizer(max_features=100))
    tp.add([remove_symbols, TfidfVectorizer(max_features=800)])

    tp.compile(
        estimator=XGBClassifier(
            objective='multi:softmax', n_estimators=100, num_class=4, learning_rate=0.075,
            colsample_bytree=0.7, subsample=0.8, eval_metric='merror'
        )
    )
    x_train, x_test, y_train, y_test = train_test_split(newsgroup_data[0], newsgroup_data[1], test_size=0.33)
    tp.run(x_train, y_train)
    y_pred = tp.predict(x_test)

    assert accuracy_score(y_test, y_pred) > 0.9
