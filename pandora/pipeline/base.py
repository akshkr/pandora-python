from abc import ABCMeta, abstractmethod
from pandora.factory import get_template


class Pipeline(metaclass=ABCMeta):
    """
    Base pipeline

    This is base pipeline for inheritance of other pipeline classes

    Parameters
    ----------
    model : str
        Type of the model

    Attributes
    ----------
    _template : object
        Template object encapsulates skeleton of the Pipeline.
        It comprises of preprocessing steps, transformer, estimator
    _data : np.ndarray
        Data Stored after preprocessing
    """

    def __init__(self, model):
        self._n_jobs = None
        self._data = None
        self._template = get_template(model)
        self.cv_params = None

    def get_data(self):
        """
        Get preprocessed features

        Returns
        -------
            preprocessed features
        """
        if self._data is None:
            print(f'No Data retained in memory. Call "pipeline.run" with "retain_data=True".')

        return self._data

    def add(self, preprocessor=None, **preprocessor_params):
        """
        Adds preprocessing Steps to pipeline

        Parameters
        ----------
        preprocessor : object or function  or list
            Preprocessor class or function of list containing Preprocessor
            or functions. These are the function(s) applied to one vector
        preprocessor_params
            column/columns
        """
        self._template.add_preprocessor(preprocessor, **preprocessor_params)

    def compile(self, transformer=None, estimator=None, **estimator_args):
        """
        Adds Transformer and Estimator to the template

        Parameters
        ----------
        transformer
        estimator : object
            Model to Estimate predictions
        estimator_args
            arguments for Estimator
        """
        self._template.add_transformer(transformer)
        self._template.add_estimator(estimator, **estimator_args)

    @abstractmethod
    def set_processor(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def enable_cv(self, *args, **kwargs):
        raise NotImplementedError

    def disable_cv(self):
        """
        Function to disable cross-validation
        """
        self.cv_params = None

    @abstractmethod
    def run(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def predict(self, *args, **kwargs):
        raise NotImplementedError
