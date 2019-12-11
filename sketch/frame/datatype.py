import datetime


class DTFrame:
	"""
	Frame to handle Data Type processing like Date/time, Amount, Location
	
	Parameters
	----------
	data : DataFrame
		The input is the DataFrame on which the operations are to be done
	"""
	
	def __init__(self, data):
		object.__setattr__(self, "_data", data)

	def datetime_delta(self, column_name):
		"""
		Uses Datetime delta column to create new features of month, week, dat, hour, day-week and day
		
		Parameters
		----------
		column_name : String
			Name of the datetime delta column
		"""
		start_date = datetime.datetime.strptime('2017-11-30', '%Y-%m-%d')

		self._data['dt'] = self._data[column_name].apply(lambda x: start_date + datetime.timedelta(seconds=x))
		
		self._data[f'{column_name}_m'] = (self._data['dt'].dt.year - 2017) * 12 + self._data['dt'].dt.month
		self._data[f'{column_name}_w'] = (self._data['dt'].dt.year - 2017) * 52 + self._data['dt'].dt.weekofyear
		self._data[f'{column_name}_d'] = (self._data['dt'].dt.year - 2017) * 365 + self._data['dt'].dt.dayofyear
		
		self._data[f'{column_name}_hour'] = self._data['dt'].dt.hour
		self._data[f'{column_name}_day_week'] = self._data['dt'].dt.dayofweek
		self._data[f'{column_name}_day'] = self._data['dt'].dt.day
		
		self._data = self._data.drop(columns=['dt'])
