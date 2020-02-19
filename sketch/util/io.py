import pickle
import os


def save_model(model, name, path):
	model_file_path = os.path.join(path, f'{name}.sav')
	pickle.dump(model, open(model_file_path, 'wb'))
	
	
def load_model(name, path):
	model_file_path = os.path.join(path, f'{name}.sav')
	loaded_model = pickle.load(open(model_file_path, 'rb'))
	return loaded_model
