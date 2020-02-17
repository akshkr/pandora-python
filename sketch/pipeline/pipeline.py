from sketch.util.dataframe import validate_column_names
from sketch.util.transformer import transform
from joblib import Parallel, delayed


class Pipeline:
	
	def __init__(self, steps):
		"""
		
		Args:
			steps:
		"""
		self._steps = steps
		self._validate_steps()

	def _validate_steps(self):
		"""
		
		Returns:

		"""
		pkey, transformers, _, _ = zip(*self._steps[:-1])
		_, estimator, _ = self._steps[-1]
		
		for t in transformers:
			if callable(t):
				pass
			else:
				if not (hasattr(t, "fit") or hasattr(t, "fit_transform")) or \
					not hasattr(t, "transform"):
					raise TypeError(
						f"All intermediate steps should be "
						f"transformers and implement fit and transform "
						f"{t} {type(t)} doesnt")
		
		if not hasattr(estimator, "fit"):
			raise TypeError(
				f"Last step of Pipeline should implement fit "
				f"{estimator} {type(estimator)} doesnt")
		
		if len(set(pkey)) != len(pkey):
			raise ValueError(
				f"Estimator names must be unique"
				f"{pkey}")
		
	def _fit(self, features, target, n_jobs):
		"""
		
		Returns:

		"""
		pkey, estimators, columns, params = zip(*self._steps[:-1])
		validate_column_names(features.columns, columns)
		
		result = Parallel(n_jobs=n_jobs, backend='threading')\
			(delayed(transform)(i for i in zip([features]*len(estimators), estimators, columns, params)))
		
	def fit(self, features, target, n_jobs=None):
		"""
		
		Returns:

		"""
		self._fit(features, target, n_jobs)
	
	def predict(self):
		pass
