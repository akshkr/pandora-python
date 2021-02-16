from .base import BaseCallback
from time import time


class PipelineCallback(BaseCallback):
    """
    Callback for Pipelines
    """
    def __init__(self):
        self._time_index = None

        self.verbose = None

    def set_params(self, params):
        self.verbose = params['verbose']

    def on_preprocess_begin(self):
        self._time_index = time()

    def on_preprocess_end(self):
        if self.verbose:
            print(f'Preprocess completed in {round(time() - self._time_index)}')

    def on_transform_begin(self):
        pass

    def on_transform_end(self):
        pass

    def on_estimation_begin(self):
        self._time_index = time()

    def on_estimation_end(self):
        if self.verbose:
            print(f'Model built in {round(time() - self._time_index)}')

    def on_prediction_begin(self):
        pass

    def on_prediction_end(self):
        pass
