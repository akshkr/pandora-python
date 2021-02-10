from tensorflow.keras.preprocessing.image import ImageDataGenerator


class Generator:

    def __init__(self, **augmentation_params):
        self.generator = ImageDataGenerator(**augmentation_params)
        self.method = None
        self.flow = None
        self.params = None

    def set_method(self, method):
        """
        Set the flow method

        Parameters
        ----------
        method : str ("df" or "dir")
            Method to load image Dataset
        """
        if method not in ['dir', 'df']:
            raise ValueError(f'Invalid input. Please enter "dir" or "df"')
        else:
            self.method = method

        if self.method == 'df':
            self.flow = self.generator.flow_from_dataframe
        elif self.method == 'dir':
            self.flow = self.generator.flow_from_directory

    def set_params(self, directory, dataframe=None, **generator_args):
        """

        Parameters
        ----------
        directory
        dataframe
        generator_args

        Returns
        -------

        """
        if self.method == 'df':
            self.params = {'directory': directory, 'dataframe': dataframe, **generator_args}
        elif self.method == 'dir':
            self.params = {'directory': directory, **generator_args}

    def generate(self, subset='training'):
        """

        Parameters
        ----------
        subset

        Returns
        -------

        """

        if subset not in ['training', 'validation']:
            raise ValueError('Invalid input. Enter "training" or "Validation"')

        return self.flow(**self.params, subset=subset)