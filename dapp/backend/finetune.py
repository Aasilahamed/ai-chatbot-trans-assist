from transformers import RobertaForSequenceClassification, RobertaTokenizer, Trainer, TrainingArguments
from datasets import Dataset
import torch

# Define custom intents and example data
data = {
    "text": [
        "I want to buy tokens", 
        "How do I buy tokens?", 
        "I want to sell my tokens", 
        "Sell 100 tokens", 
        "How do I transfer tokens?", 
        "Can you help me stake my tokens?"
    ],
    "intent": [
        "buy", 
        "buy", 
        "sell", 
        "sell", 
        "transfer", 
        "stake"
    ]
}

# Convert to Dataset format
dataset = Dataset.from_dict(data)

# Load tokenizer and model
model_name = "roberta-base"
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=4)  # Adjust number of labels

# Tokenize the data
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# Tokenize the dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Set training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    weight_decay=0.01,
    logging_dir="./logs",
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")
