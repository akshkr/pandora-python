import pandas as pd
import numpy as np

from pandora.util.statistics import Frame, NumpyFrame, Dataframe


class StatisticalPreprocessor:
    """
    Factory for Statistical Preprocessor operating on various DataTypes

    Calling this class returns the preprocessor of relevant datatype
    passed in the constructor
    """

    def __new__(cls, frame):
        if isinstance(frame, pd.DataFrame):
            return Dataframe(frame)

        elif isinstance(frame, np.ndarray):
            return NumpyFrame(frame)

        else:
            return Frame(frame)
