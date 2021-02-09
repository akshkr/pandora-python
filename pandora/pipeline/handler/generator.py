def compile_generator(augmentation_params=None):
    if augmentation_params is None:
        augmentation_params = {}

    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    generator = ImageDataGenerator(**augmentation_params)

    return generator

# if method == 'df':
#     generator.flow_from_dataframe(dataframe, directory, **generator_args)
# elif method == 'dir':
#     generator.flow_from_directory(directory, **generator_args)
