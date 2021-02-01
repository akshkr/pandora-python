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
    """

    def __init__(self, model):

        self._n_jobs = 1
        self._features = None
        self._template = get_template(model)

    def get_data(self):
        """
        Get preprocessed features

        Returns
        -------
            preprocessed features
        """
        if self._features is None:
            print(f'No features retained in memory. Call "pipeline.run" with "retain_features=True".')

        return self._features

    def add(self, preprocessor=None, **kwargs):
        """
        Adds preprocessing Steps to pipeline

        Parameters
        ----------
        preprocessor : object or function  or list
            Preprocessor class or function of list containing Preprocessor
            or functions. These are the function(s) applied to one vector
        kwargs
            column/columns
        """
        self._template.add_preprocessor(preprocessor, **kwargs)

    def compile(self, transformer=None, estimator=None, **kwargs):
        """
        Adds Transformer and Estimator to the template

        Parameters
        ----------
        transformer
        estimator : object
            Model to Estimate predictions
        kwargs
            arguments for Estimator
        """
        self._template.add_transformer(transformer)
        self._template.add_estimator(estimator, **kwargs)

    @abstractmethod
    def run(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def predict(self, *args, **kwargs):
        raise NotImplementedError
