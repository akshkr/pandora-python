from sklearn.model_selection import KFold


def kfold_validation(n_splits, model_class, model_args, train_df, target, accuracy_check):
	"""
	Make Prediction
	
	Parameters
	----------
	n_splits : int
		Number of folds for validation
	model_class : Class
		Class of model
	model_args : dict
		Argument for model
	train_df : DataFrame
		Training dataset
	target : Series
		Training target
	accuracy_check : Function to check the accuracy
	"""
	
	folds = KFold(n_splits=n_splits, shuffle=True, random_state=42)
	
	for fold, (train_idx, validation_idx) in enumerate(folds.split(train_df, target)):
		
		train_x = train_df.iloc[train_idx, :]
		train_y = target[train_idx]
		validation_x = train_df.iloc[validation_idx, :]
		validation_y = target[validation_idx]
		
		model = model_class(**model_args)
		model.fit(train_x, train_y, **model_args)
		validation_prediction = model.predict(validation_x)
		
		accuracy_check(validation_y, validation_prediction)
		del model
	
	# Return the model after training on the entire data
	model = model_class(**model_args)
	model.fit(train_df, target)
	return model
