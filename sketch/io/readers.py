import yaml


def read_yaml(file_path=None):
	"""
	Method to read YAML file
	
	:param file_path: Input file path
	:return: Yaml object / Dictionary
	"""
	
	try:
		with open(file_path, 'r') as fp:
			return yaml.safe_load(fp)
		
	except Exception as ex:
		raise ex
