import time
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

model = DistilBertForSequenceClassification.from_pretrained("./model")
tokenizer = DistilBertTokenizerFast.from_pretrained("./model")

model.eval()

text = "I was charged twice for my subscription"

start = time.time()

inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

end = time.time()

latency = end - start

print("Latency:", latency, "seconds")

assert latency < 0.5, "Latency exceeds 500ms!"

print("Latency test passed ")