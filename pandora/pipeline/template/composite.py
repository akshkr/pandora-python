from .base import Template


class CompositeTemplate(Template):
    """
    Template for Composite Pipeline

    This template is used for Composite type dataset based pipeline
    """
    def __init__(self):
        super().__init__()

    def add_preprocessor(self, preprocessor, column=None):
        """
        Adds preprocessor to the template

        Parameters
        ----------
        preprocessor
            Preprocessor to be used on the input data
        column : str, optional
            Column name if passed data is of type DataFrame
        """
        self.preprocessing_steps.append({'preprocessor': preprocessor, 'column': column})
