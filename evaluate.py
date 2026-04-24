import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Load model
model = DistilBertForSequenceClassification.from_pretrained("./model")
tokenizer = DistilBertTokenizerFast.from_pretrained("./model")

model.eval()

# Test dataset (manual)
texts = [
    "I was charged twice",
    "App is not working",
    "Please add export feature",
    "Worst service ever",
    "Just checking something"
]

labels = [0, 1, 2, 3, 4]

inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

preds = torch.argmax(outputs.logits, dim=1).numpy()

# Metrics
acc = accuracy_score(labels, preds)
f1 = f1_score(labels, preds, average="weighted")
cm = confusion_matrix(labels, preds)

print("\nAccuracy:", acc)
print("F1 Score:", f1)
print("\nConfusion Matrix:\n", cm)