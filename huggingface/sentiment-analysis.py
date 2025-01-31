# https://www.youtube.com/watch?v=QEaBAZQCtwE&t=239s
from transformers import pipeline
classifier = pipeline("sentiment-analysis")
res = classifier("This ML stuff's really getting exciting huh")
print(res)
