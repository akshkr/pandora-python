from abc import ABCMeta, abstractmethod


class Pipeline(metaclass=ABCMeta):

	@abstractmethod
	def add(self, *args, **kwargs):
		raise NotImplementedError

	@abstractmethod
	def compile(self, *args, **kwargs):
		raise NotImplementedError

	@abstractmethod
	def run(self, *args, **kwargs):
		raise NotImplementedError

	@abstractmethod
	def predict(self, *args, **kwargs):
		raise NotImplementedError
