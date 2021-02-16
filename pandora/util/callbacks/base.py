class BaseCallback:

    def set_params(self, *args, **kwargs):
        raise NotImplementedError

    def on_preprocess_begin(self, *args, **kwargs):
        raise NotImplementedError

    def on_preprocess_end(self, *args, **kwargs):
        raise NotImplementedError

    def on_transform_begin(self, *args, **kwargs):
        raise NotImplementedError

    def on_transform_end(self, *args, **kwargs):
        raise NotImplementedError

    def on_estimation_begin(self, *args, **kwargs):
        raise NotImplementedError

    def on_estimation_end(self, *args, **kwargs):
        raise NotImplementedError

    def on_prediction_begin(self, *args, **kwargs):
        raise NotImplementedError

    def on_prediction_end(self, *args, **kwargs):
        raise NotImplementedError
