import torch
from datasets import Dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split

# Sample dataset (you can expand)
data = [
    {"text": "I was charged twice", "label": 0},
    {"text": "App crashes on login", "label": 1},
    {"text": "Add dark mode feature", "label": 2},
    {"text": "Very bad experience", "label": 3},
    {"text": "General question", "label": 4},
] * 200  # simulate 1000 samples

label_map = {
    0: "billing",
    1: "technical_issue",
    2: "feature_request",
    3: "complaint",
    4: "other"
}

train_data, val_data = train_test_split(data, test_size=0.2)

train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length")

train_dataset = train_dataset.map(tokenize)
val_dataset = val_dataset.map(tokenize)

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=5)

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    logging_dir="./logs",
    save_strategy="no"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

trainer.train()

model.save_pretrained("./model")
tokenizer.save_pretrained("./model")

print("Training complete. Model saved.")