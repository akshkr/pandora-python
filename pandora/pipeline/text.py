from .handler import handle_train_preprocessor
from ..util.process import parallelize
from ..factory import get_model
from .base import Pipeline


class TextPipeline(Pipeline):
    def __init__(self, model=None):
        model = 'text' if model is None else model
        self.model = get_model(model)

    def _extract_feature_array(self, data):
        preprocessors = [x['preprocessor'] for x in self.model.preprocessing_steps]
        features = [x['column'] for x in self.model.preprocessing_steps]
        features = [data[col] if col is not None else data for col in features]

        return preprocessors, features

    def add(self, preprocessor, **kwargs):
        self.model.add_preprocessor(preprocessor, **kwargs)

    def compile(self, transformer, estimator):
        self.model.add_transformer(transformer)
        self.model.add_estimator(estimator)

    def run(self, data):
        preprocessors, features = self._extract_feature_array(data)
        parallelize(
            handle_train_preprocessor,
            [preprocessors, features],
            n_jobs=4
        )

    def predict(self, *args, **kwargs):
        pass
