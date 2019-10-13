from sklearn.model_selection import KFold
from sketch.io.readers import read_yaml
from sketch.models.model import Model
from inspect import getsourcefile
import lightgbm as lgb
import numpy as np
import os
import gc


class Baseline(Model):
	
	def __init__(self, can):
		"""
		:param can: Canvas object
		"""
		self._can = can
		self._estimator = None
		
	def make_prediction(self):
		"""
		Makes prediction
		"""
		folds = KFold(n_splits=10, shuffle=True, random_state=42)
		
		predictions = np.zeros(len(self._can.test))
		for folds, (train_idx, validation_idx) in enumerate(folds.split(self._can.train, self._can.target)):
			print(f'Fold: {folds}')
			
			train_x = self._can.train.iloc[train_idx, :]
			train_y = self._can.target[train_idx]
			
			validation_x = self._can.train.iloc[validation_idx, :]
			validation_y = self._can.target[validation_idx]
			
			train_data = lgb.Dataset(train_x, label=train_y)
			validation_data = lgb.Dataset(validation_x, label=validation_y)
			del train_x, train_y, validation_x, validation_y
			
			file_source = os.path.split(os.path.abspath(getsourcefile(lambda: 0)))[0]
			estimator = lgb.train(
				read_yaml(os.path.join(file_source, 'params/parameter.yml')).get('lgb_params'),
				train_data,
				valid_sets=[train_data, validation_data],
				verbose_eval=200
			)
			
			if self._can.test is not None:
				this_prediction = estimator.predict(self._can.test)
				predictions += this_prediction/10
			
			del train_data, validation_data
			
			gc.collect()
	
		return predictions
