class NumeralData:
    def __init__(self):
        self.raw_data = None
        self.data_matrix = None
        self.target = None

    def update(self, raw_data=None, data_matrix=None, target=None):
        if raw_data:
            self.raw_data = raw_data
        if data_matrix:
            self.data_matrix = data_matrix
        if target:
            self.target = target
