def compile_generator(template):
    """
    Creates Generator object from Image pipeline template

    Parameters
    ----------
    template
        Image Pipeline template

    Returns
    -------
        Generator object
    """

    from pandora.util.io.image import Generator

    # Aggregate Generator constructor parameters
    # Handle validation on generator
    generator_args = {**template.augmentation_params}
    if template.cross_val:
        generator_args['validation_split'] = template.cross_val['validation_split']
    generator = Generator(**generator_args)

    # Set Flow parameters
    generator.set_params(
        template.generator_params['method'],
        template.generator_params['directory'],
        template.generator_params['target_size'],
        template.generator_params['dataframe'],
        **template.generator_params['generator_params']
    )

    return generator
