from pandora.util.statistics import Frame, NumpyFrame, Dataframe
import pandas as pd
import numpy as np


class StatisticalPreprocessor:

    def __new__(cls, frame):
        if isinstance(frame, pd.DataFrame):
            return Dataframe(frame)

        elif isinstance(frame, np.ndarray):
            return NumpyFrame(frame)

        else:
            return Frame(frame)
