import pickle
import os


def save_model(model, name, path):
    """
    Saves model to a pickle file

    Args:
        model (object): Model
        name (str): model file name
        path: path to save model
    """
    model_file_path = os.path.join(path, f'{name}.sav')
    pickle.dump(model, open(model_file_path, 'wb'))


def load_model(name, path):
    """
    Loads model from a path

    Args:
        name (str): Name of model file
        path (str): Path in which model is saved

    Returns:
        Loaded model
    """
    model_file_path = os.path.join(path, f'{name}.sav')
    loaded_model = pickle.load(open(model_file_path, 'rb'))
    return loaded_model
