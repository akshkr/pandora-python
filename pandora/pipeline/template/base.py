from abc import abstractmethod


class BaseTemplate:
    """
    Base Template class for inheritance

    This class must be inherited by all child Template class

    Attributes
    ----------
    preprocessing_steps : list
        list of preprocessing step details
    transformer
    estimator
        Estimator
    estimator_args
        Arguments passes in Estimator
    """

    def __init__(self):
        self.preprocessing_steps = list()
        self.transformer = None
        self.estimator = None
        self.estimator_args = None

    @abstractmethod
    def add_preprocessor(self, *args, **kwargs):
        raise NotImplementedError

    def add_transformer(self, transformer):
        """
        Adds transformer to the template

        Parameters
        ----------
        transformer
        """
        self.transformer = transformer

    def add_estimator(self, estimator, **estimator_args):
        """
        Adds Estimator and its arguments in template

        Parameters
        ----------
        estimator
        estimator_args
            Estimator Arguments while fitting (not the hyper-parameters)
        """
        self.estimator = estimator
        self.estimator_args = estimator_args
