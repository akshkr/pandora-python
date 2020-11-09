from ..core.pipeline_template import TextTemplate, ImageTemplate


def get_template(variant=None):
    if variant == 'text':
        return TextTemplate()

    if variant == 'image':
        return ImageTemplate()

    raise ValueError
