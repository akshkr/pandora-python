from ..models import TextModel, ImageModel


def get_model(variant=None):
    if variant == 'text':
        return TextModel()

    elif variant == 'image':
        return ImageModel()

    else:
        raise ValueError
