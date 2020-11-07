from ..core.pipeliine_template import TextTemplate, ImageTemplate


def get_template(variant=None):
    if variant == 'text':
        return TextTemplate()

    elif variant == 'image':
        return ImageTemplate()

    else:
        raise ValueError
