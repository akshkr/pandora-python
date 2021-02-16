from pandora.preprocessing import StatisticsMeasure
from pandora.util import seed_everything

import numpy as np

seed_everything()


def test_statistical_preprocessor():
    test_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 10], [9, 11, 12]])

    preprocessor_obj = StatisticsMeasure()
    output = preprocessor_obj.fit_transform(test_array)

    real_output = np.array(
        [
            [6, 15, 25, 32],
            [2, 5, 8.333, 10.666],
            [2, 5, 8, 11],
            [0.82, 0.816, 1.247, 1.247],
            [0, 0, 0.381, -0.381],
            [-1.5, -1.5, -1.5, -1.5]
        ]
    )
    np.testing.assert_array_equal(real_output.round(2).T, output.round(2))
