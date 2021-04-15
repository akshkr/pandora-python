from .base import BaseTransformer


class ImageScaler(BaseTransformer):
    """
    Scales image array
    """
    def fit_transform(self, features):
        images = features.astype('float32')
        images /= 255.0

        return images
