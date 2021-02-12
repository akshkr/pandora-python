from tensorflow.keras.preprocessing.image import ImageDataGenerator


class Generator:
    """
    Generates batches of image data using augmentation
    """

    def __init__(self, **augmentation_params):
        self.generator = ImageDataGenerator(**augmentation_params)
        self.method = None
        self.params = None

    def set_params(self, method, directory, target_size, dataframe=None, **generator_args):
        """
        Set parameters for Generator

        Parameters
        ----------
        method : str ("df" or "dir")
            The method of flow
        directory : str
            Directory containing images
        target_size : tuple
            Output size of images to be rescaled to
        dataframe : DataFrame
            DataFrame with image path and labels
        generator_args
            Arguments to be passed to generator flow function
        """
        self.method = method
        self.params = {'directory': directory, 'target_size': target_size, **generator_args}

        if self.method == 'df':
            self.params['dataframe'] = dataframe

    def generate(self, subset='training'):
        """
        Generated training and validation data

        Parameters
        ----------
        subset : str
            Which batch to exhaust

        Returns
        -------
            Image and target variable
        """

        if subset not in ['training', 'validation']:
            raise ValueError('Invalid input. Enter "training" or "Validation"')

        if self.method == 'df':
            return self.generator.flow_from_dataframe(**self.params, subset=subset)
        elif self.method == 'dir':
            return self.generator.flow_from_directory(**self.params, subset=subset)
        else:
            raise ValueError(f'Invalid input. Please enter "dir" or "df"')
