from sketch.core.column import group_column_operate


class RTFrame:
		
	def group_target_aggregate(self, columns):
		"""
		Groups and aggregate target mean on columns
		"""
		for column in columns:
			self._data = group_column_operate(
				self.train_df, self._data, self.target_column_name,
				column, 'mean'
			)
