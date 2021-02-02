from pandora.pipeline.template import TabularTemplate, ImageTemplate


def get_template(variant=None):
    """
    Returns template for the required variant

    Parameters
    ----------
    variant : str
        Variant for which template object is needed

    Returns
    -------
        Template object
    """
    if variant == 'tabular':
        return TabularTemplate()

    if variant == 'image':
        return ImageTemplate()

    raise ValueError
