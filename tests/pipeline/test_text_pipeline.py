from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

from pandora.util.io import save_model, load_model
from .._functions import remove_symbols
from pandora import TextPipeline


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


multiclass_newsgroup_data = newsgroup_data(4)


def test_single_preprocessor():
    tp = TextPipeline()
    tp.add(TfidfVectorizer(max_features=1000))
    tp.compile(
        estimator=XGBClassifier(
            objective='multi:softmax', eval_metric='merror'
        )
    )

    x_train, x_test, y_train, y_test = train_test_split(multiclass_newsgroup_data[0], multiclass_newsgroup_data[1], test_size=0.33)
    tp.run(x_train, y_train)
    y_pred = tp.predict(x_test)

    assert accuracy_score(y_test, y_pred) > 0.9


def test_multi_preprocessor():
    tp = TextPipeline()
    tp.add(TfidfVectorizer(max_features=1000))
    tp.add(TfidfVectorizer(max_features=20))
    tp.compile(
        estimator=XGBClassifier(
            objective='multi:softmax', eval_metric='merror'
        )
    )

    x_train, x_test, y_train, y_test = train_test_split(multiclass_newsgroup_data[0], multiclass_newsgroup_data[1], test_size=0.33)
    tp.run(x_train, y_train)
    y_pred = tp.predict(x_test)

    assert accuracy_score(y_test, y_pred) > 0.9


def test_multi_preprocessor_with_func():
    tp = TextPipeline()
    tp.add(TfidfVectorizer(max_features=100))
    tp.add([remove_symbols, TfidfVectorizer(max_features=800)])
    tp.compile(
        estimator=XGBClassifier(
            objective='multi:softmax', eval_metric='merror'
        )
    )

    x_train, x_test, y_train, y_test = train_test_split(multiclass_newsgroup_data[0], multiclass_newsgroup_data[1], test_size=0.33)
    tp.run(x_train, y_train)
    y_pred = tp.predict(x_test)

    assert accuracy_score(y_test, y_pred) > 0.9


def test_multi_preprocessor_with_dump(temp_dir):
    tp = TextPipeline()
    tp.add(TfidfVectorizer(max_features=100))
    tp.add([remove_symbols, TfidfVectorizer(max_features=800)])
    tp.compile(
        estimator=XGBClassifier(
            objective='multi:softmax', eval_metric='merror'
        )
    )

    x_train, x_test, y_train, y_test = train_test_split(multiclass_newsgroup_data[0], multiclass_newsgroup_data[1], test_size=0.33)
    tp.run(x_train, y_train)
    save_model(tp, 'model', temp_dir)
    del tp

    tp2 = load_model('model', temp_dir)
    y_pred = tp2.predict(x_test)

    assert accuracy_score(y_test, y_pred) > 0.9
