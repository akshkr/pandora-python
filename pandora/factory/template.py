from pandora.pipeline.template import TabularTemplate, ImageTemplate
from pandora.reference.pipeline import PipelineTypes


def get_template(variant):
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
    if variant == PipelineTypes.TABULAR.value:
        return TabularTemplate()

    if variant == PipelineTypes.IMAGE.value:
        return ImageTemplate()

    raise ValueError(f'Undefined template variant "{variant}"')
