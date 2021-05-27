def check_key(key, dictionary):
    if key not in dictionary.keys():
        raise KeyError(f'{key} not in space. Please use one of the values: {dictionary.keys()}')
