from transformers import pipeline
classifier = pipeline("zero-shot-classification")
res = classifier(
  "This is a demand for better food and wages",
  candidate_labels=["ad", "politics", "business", "diary"]
)
print (res)
