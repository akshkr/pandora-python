from .handler import handle_transformer, validate_transformer, handle_estimator
from sketch.models.accuracy_score import binary_classification_accuracy
from sketch.util.dataframe import validate_column_names
from sketch.util.datatype import convert_to_numpy
from joblib import Parallel, delayed
import numpy as np


class Pipeline:
	
	def __init__(self, steps):
		"""
		
		Args:
			steps:
		"""
		self._steps = steps
		self._validate_steps()
		self.model = dict()

	def _validate_steps(self):
		"""
		
		Returns:

		"""
		pkey, transformers, _, _ = zip(*self._steps[:-1])
		_, estimator, _ = self._steps[-1]
		
		validate_transformer(transformers)
		
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
		
		transformed_features = Parallel(n_jobs=n_jobs, backend='multiprocessing')\
			(delayed(handle_transformer)(*i) for i in zip(
				[self]*len(estimators), [features]*len(estimators), pkey, estimators, columns, params))
		
		transformed_features = convert_to_numpy(transformed_features)
		# print([x.shape[1] for x in transformed_features])
		transformed_features = np.hstack(transformed_features)
		pkey, estimator, param = self._steps[-1]
		
		# self._model = handle_estimator(
		# 	self, estimator, pkey, transformed_features, target, params, binary_classification_accuracy)
		
	def fit(self, features, target, n_jobs=None):
		"""
		
		Returns:

		"""
		self._fit(features, target, n_jobs)
		# return result
	
	def predict(self):
		pass
