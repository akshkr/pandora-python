from sklearn.ensemble import (
	RandomForestClassifier, AdaBoostClassifier,
	GradientBoostingClassifier, ExtraTreesClassifier
)
from sklearn.model_selection import StratifiedKFold
from threading import Thread
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import time


def stack(canvas_obj):
	"""
	
	:param canvas_obj:
	:return:
	"""
	stack_obj = _Stack(canvas_obj)
	stack_obj.make_stack()


class _Stack:
	
	def __init__(self, canvas):
		self._canvas = canvas
		self.n_train = canvas.train.shape[0]
		self.n_test = canvas.test.shape[0]
		self.seed = 49  # for reproducibility
		self.n_folds = 5  # set folds for out-of-fold prediction
		self.kf = StratifiedKFold(n_splits=self.n_folds, random_state=self.seed)
		
	def get_oof(self, clf, output_list, tri, tei):
		time_idx = time.time()
		oof_train = np.zeros((self.n_train,))
		oof_test = np.zeros((self.n_test,))
		oof_test_skf = np.empty((self.n_folds, self.n_test))
		
		i = 0
		for train_index, test_index in self.kf.split(self._canvas.train, self._canvas.target):
			x_tr = self._canvas.train.loc[train_index]
			y_tr = self._canvas.target.loc[train_index]
			x_te = self._canvas.train.loc[test_index]
			
			clf.train(x_tr, y_tr)
	
			oof_train[test_index] = clf.predict(x_te)
			oof_test_skf[i, :] = clf.predict(self._canvas.test)
			i += 1
	
		oof_test[:] = oof_test_skf.mean(axis=0)
		
		print(f'Time for {clf} : {time.time() - time_idx}')
		output_list[tri] = oof_train.reshape(-1, 1)
		output_list[tei] = oof_test.reshape(-1, 1)
	
	def make_stack(self):
		"""
		
		:return:
		"""
		rf_params = {
			'n_jobs': -1,
			'n_estimators': 500,
			'warm_start': True,
			# 'max_features': 0.2,
			'max_depth': 6,
			'min_samples_leaf': 2,
			'max_features': 'sqrt',
			'verbose': 0
		}
		
		# Extra Trees Parameters
		et_params = {
			'n_jobs': -1,
			'n_estimators': 100,
			# 'max_features': 0.5,
			'max_depth': 8,
			'min_samples_leaf': 2,
			'verbose': 0
		}
		
		# AdaBoost parameters
		ada_params = {
			'n_estimators': 100,
			'learning_rate': 0.75
		}
		
		# Gradient Boosting parameters
		# gb_params = {
		# 	'n_estimators': 100,
		# 	# 'max_features': 0.2,
		# 	'max_depth': 5,
		# 	'min_samples_leaf': 2,
		# 	'verbose': 0
		# }
		
		# Support Vector Classifier parameters
		# svc_params = {
		# 	'kernel': 'linear',
		# 	'C': 0.025
		# }
		
		output_list = [0, 0, 0, 0, 0, 0]
		rf = _ModelHelper(model=RandomForestClassifier, seed=self.seed, params=rf_params)
		t1 = Thread(target=self.get_oof, args=(rf, output_list, 0, 1))
		# rf_oof_train, rf_oof_test = self.get_oof(rf)  # Random Forest
		
		et = _ModelHelper(model=ExtraTreesClassifier, seed=self.seed, params=et_params)
		t2 = Thread(target=self.get_oof, args=(et, output_list, 2, 3))
		# et_oof_train, et_oof_test = self.get_oof(et)  # Extra Trees
		
		ada = _ModelHelper(model=AdaBoostClassifier, seed=self.seed, params=ada_params)
		t3 = Thread(target=self.get_oof, args=(ada, output_list, 4, 5))
		
		t1.start()
		t2.start()
		t3.start()
		
		t1.join()
		t2.join()
		t3.join()
		
		# gb = _ModelHelper(model=GradientBoostingClassifier, seed=self.seed, params=gb_params)
		# gb_oof_train, gb_oof_test = self.get_oof(gb)  # Gradient Boost
		
		# svc = _ModelHelper(model=SVC, seed=self.seed, params=svc_params)
		# svc_oof_train, svc_oof_test = self.get_oof(svc)  # Support Vector Classifier
		
		base_predictions_train = pd.DataFrame({
			'RandomForest': output_list[0].ravel(),
			'ExtraTrees': output_list[2].ravel(),
			'AdaBoost': output_list[4].ravel(),
			# 'GradientBoost': gb_oof_train.ravel()
		})
		
		base_predictions_test = pd.DataFrame({
			'RandomForest': output_list[1].ravel(),
			'ExtraTrees': output_list[3].ravel(),
			'AdaBoost': output_list[5].ravel(),
			# 'GradientBoost': gb_oof_test.ravel()
		})
		
		stack_df = pd.concat([base_predictions_train, base_predictions_test], axis=0).reset_index(drop=True)
		self._canvas._data = pd.concat([self._canvas.dataframe.copy(), stack_df], axis=1)
		
		
class _ModelHelper:
	
	def __init__(self, model, seed=0, params=None):
		params['random_state'] = seed
		self.model = model(**params)
	
	def train(self, x_train, y_train):
		self.model.fit(x_train, y_train)
	
	def predict(self, x):
		return self.model.predict(x)
	
	def fit(self, x, y):
		return self.model.fit(x, y)
	
	def feature_importance(self, x, y):
		print(self.model.fit(x, y).feature_importances_)
