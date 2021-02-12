from tensorflow.keras.preprocessing.image import ImageDataGenerator


class Generator:

    def __init__(self, **augmentation_params):
        self.generator = ImageDataGenerator(**augmentation_params)
        self.method = None
        self.flow = None
        self.params = None

    def set_params(self, method, directory, target_size, dataframe=None, **generator_args):
        """

        Parameters
        ----------
        method
        directory
        target_size
        dataframe
        generator_args

        Returns
        -------

        """
        self.method = method
        self.params = {'directory': directory, 'target_size': target_size, **generator_args}

        if self.method == 'df':
            self.params['dataframe'] = dataframe

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

        if self.method == 'df':
            return self.generator.flow_from_dataframe(**self.params, subset=subset)
        elif self.method == 'dir':
            return self.generator.flow_from_directory(**self.params, subset=subset)
        else:
            raise ValueError(f'Invalid input. Please enter "dir" or "df"')
