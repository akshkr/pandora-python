from pandora.pipeline.template import CompositeTemplate, ImageTemplate


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
    if variant == 'composite':
        return CompositeTemplate()

    if variant == 'image':
        return ImageTemplate()

    raise ValueError
