from .base import BaseTransform


class ImageScaler(BaseTransform):
    """
    Scales image array
    """
    def fit_transform(self, features):
        images = features.astype('float32')
        images /= 255.0

        return images
