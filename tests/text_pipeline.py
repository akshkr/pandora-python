from sklearn.datasets import fetch_20newsgroups
from pandora import TextPipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

data = fetch_20newsgroups(return_X_y=True)
text_model = TextPipeline()
text_model.add(TfidfVectorizer())
text_model.add(TfidfVectorizer())

op = text_model.run(data[0], data[1])

print(op)
# print(list(op[0][0]))
