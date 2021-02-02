from .frame import Frame


class NumpyFrame(Frame):
    """
    Statistical operation class for Numpy arrays
    """

    def __init__(self, frame):
        super().__init__(frame)
