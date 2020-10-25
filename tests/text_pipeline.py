from sklearn.datasets import fetch_20newsgroups
from pandora import TextPipeline

#
# class TestTextPipeline:
#
#     def


data = fetch_20newsgroups(return_X_y=True)

text_model = TextPipeline()
