from ..core.datastore import NumeralData, ImageData


def get_datastore(variant=None):
    if variant == 'numeral':
        return NumeralData()

    elif variant == 'image':
        return ImageData()

    else:
        raise ValueError
