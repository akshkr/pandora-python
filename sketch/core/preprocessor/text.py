def break_into_substrings(d, substring_length=3):
	d = d.replace(' ', '_')
	d = f'__{d}__'
	
	return " ".join([d[i:i + substring_length] for i in range(len(d) - substring_length + 1)])
