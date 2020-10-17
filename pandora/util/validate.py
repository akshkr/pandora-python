from sklearn.model_selection import KFold
from joblib import Parallel, delayed


def train_and_validate(model_obj, features, target, train_idx, validation_idx, accuracy_check):
	model = model_obj
	model.run(features[train_idx, :], target[train_idx])
	validation_prediction = model.predict(features[validation_idx, :])
	accuracy_check(target[validation_idx], validation_prediction)
	del model
	

def kfold_validation(n_splits, model_obj, model_args, features, target, accuracy_check):
	"""
	Make Prediction
	
	Parameters
	----------
	n_splits : int
		Number of folds for validation
	model_obj : Class
		Class of model
	model_args : dict
		Argument for model
	features : DataFrame
		Training dataset
	target : Series
		Training target
	accuracy_check : Function to check the accuracy
	"""
	
	folds = KFold(n_splits=n_splits, shuffle=True, random_state=42)

	train_index_list = list()
	validation_index_list = list()

	for fold, (train_idx, validation_idx) in enumerate(folds.split(features, target)):
		train_index_list.append(train_idx)
		validation_index_list.append(validation_idx)
		
	Parallel(n_jobs=n_splits, backend='multiprocessing') \
		(delayed(train_and_validate)(*i) for i in zip(
			[model_obj] * n_splits, [features] * n_splits, [target] * n_splits,
			train_index_list, validation_index_list, [accuracy_check] * n_splits
		))
