from sklearn.model_selection import KFold
from sketch.models.model import Model
import lightgbm as lgb
import numpy as np
import gc


lgb_params = {
	'objective': 'binary',
	'boosting_type': 'gbdt',
	'metric': 'auc',
	'n_jobs': -1,
	'learning_rate': 0.01,
	'num_leaves': 2**8,
	'max_depth': -1,
	'tree_learner': 'serial',
	'colsample_bytree': 0.7,
	'subsample_freq': 1,
	'subsample': 1,
	'n_estimators': 800,
	'max_bin': 255,
	'verbose': -1,
	'seed': 42,
	'early_stopping_rounds': 100,
}


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
		folds = KFold(n_splits=2, shuffle=True, random_state=42)
		
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
			
			estimator = lgb.train(
				lgb_params,
				train_data,
				valid_sets=[train_data, validation_data],
				verbose_eval=200
			)
			
			this_prediction = estimator.predict(self._can.test)
			predictions += this_prediction/2
			
			del train_data, validation_data
			
			gc.collect()
	
		return predictions
