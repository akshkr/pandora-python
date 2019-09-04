from sketch.util.analyser import analyse
import pandas as pd
import numpy as np


def test_analyse():
	
	a = np.random.randint(low=1, high=3, size=50)
	b = np.random.randint(low=1, high=3, size=50)
	c = np.random.randint(low=1, high=3, size=50)
	d = np.random.randint(low=1, high=3, size=50)
	e = np.random.randint(low=1, high=3, size=50)
	f = np.random.randint(low=1, high=3, size=50)
	
	total = pd.DataFrame({
		'A_1': a,
		'A_2': b,
		'A_3': c,
		'B1': d,
		'B2': e,
		'C1': f
	})
	assert analyse(total) == 1
