from ..core.pipeline_template import CompositeTemplate, ImageTemplate


def get_template(variant=None):
    if variant == 'composite':
        return CompositeTemplate()

    if variant == 'image':
        return ImageTemplate()

    raise ValueError
