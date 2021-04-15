from .base import BaseTemplate


class TabularTemplate(BaseTemplate):
    """
    Template for Composite Pipeline

    This template is used for Composite type dataset based pipeline
    """
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
