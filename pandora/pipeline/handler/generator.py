def compile_generator(template):

    from pandora.util.io.image import Generator

    generator_args = {**template.augmentation_params}
    if template.cross_val:
        generator_args['validation_split'] = template.cross_val

    generator = Generator(**generator_args)
    generator.set_method(template.generator_params['method'])
    generator.set_params(
        template.generator_params['directory'],
        template.generator_params['dataframe'],
        **template.generator_params['generator_params']
    )

    return generator
