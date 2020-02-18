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
			steps (list): List of tuples containing transformers and estimators
			Each tuple contains (pkey, estimator/transformer, column_name, params)
		"""
		self._steps = steps
		self.model = dict()
		
		# Validate the transformers and estimators
		self._validate_steps()

	def _validate_steps(self):
		pkey, transformers, _, _ = zip(*self._steps[:-1])
		_, estimator, _ = self._steps[-1]
		
		# Validate transformer
		validate_transformer(transformers)
		
		# Validate estimator
		if not hasattr(estimator, "fit"):
			raise TypeError(
				f"Last step of Pipeline should implement fit "
				f"{estimator} {type(estimator)} doesnt")
		
		# Validate primary key
		if len(set(pkey)) != len(pkey):
			raise ValueError(
				f"Estimator names must be unique"
				f"{pkey}")
		
	def _fit(self, features, target, n_jobs):
		pkey, estimators, columns, params = zip(*self._steps[:-1])
		
		# Check for columns in the DataFrame
		validate_column_names(features.columns, columns)
		
		# Transform features using parallelization
		transformed_features = Parallel(n_jobs=n_jobs, backend='multiprocessing')\
			(delayed(handle_transformer)(*i) for i in zip(
				[self]*len(estimators), [features]*len(estimators), pkey,
				estimators, columns, params))
		
		# Convert every feature to numpy array and concatenate
		transformed_features = convert_to_numpy(transformed_features)
		transformed_features = np.hstack(transformed_features)
		
		# Use estimator
		pkey, estimator, param = self._steps[-1]
		self._model = handle_estimator(
			self, estimator, pkey, transformed_features, target, params,
			binary_classification_accuracy)
		
	def fit(self, features, target, n_jobs=None):
		"""
		Train the model using the transformations and estimator

		Args:
			features (pd.DataFrame): Independent variable of training data
			target (pd.Series): Dependent variable of training data
			n_jobs (int): Number of jobs to parallelize transformations

		Returns:

		"""
		self._fit(features, target, n_jobs)
		# return result
	
	def predict(self):
		pass
