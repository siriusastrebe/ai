# https://www.youtube.com/watch?v=QEaBAZQCtwE&t=239s

from transformers import pipeline
generator = pipeline("text-generation")

res = generator(
  "An icy shiver went down his veins. He looked in front of him and saw",
  max_length=300,
  num_return_sequences=2,
)
print(res)
