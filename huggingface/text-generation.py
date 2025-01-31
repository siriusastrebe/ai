# https://www.youtube.com/watch?v=QEaBAZQCtwE&t=239s

from transformers import pipeline
generator = pipeline("text-generation", model="distilgpt2")

res = generator(
  "In this course, we will teach you how to",
  max_length=300,
  num_return_sequences=2,
)
print(res)
