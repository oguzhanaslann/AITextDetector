from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import load_from_disk

# Define label mappings
label2id = {"HUMAN": 0, "AI": 1}
id2label = {0: "HUMAN", 1: "AI"}

def map_labels_of_dataframe(frame):
    frame["label"] = frame["label"].map(label2id)
    return frame

# Load model and tokenizer
model_name = "roberta-base"
tokenizer = RobertaTokenizerFast.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# Load pre-tokenized datasets
tokenized_training_dataset = load_from_disk("data/tokenized_training")
tokenized_validation_dataset = load_from_disk("data/tokenized_validation")

# Initialize the model for sequence classification
model = RobertaForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(label2id),
    label2id=label2id,
    id2label=id2label
)

# Define training arguments
output_dir = "./full_finetune_results"
training_args = TrainingArguments(
    output_dir=output_dir,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir=f'{output_dir}/logs',
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    learning_rate=2e-5,  # Standard learning rate for full fine-tuning
    report_to="none"  # Disable wandb logging
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_training_dataset,
    eval_dataset=tokenized_validation_dataset
)

# Train the model
trainer.train()

# Save the model and tokenizer
model_output_dir = "./finetuned_roberta_full"
model.save_pretrained(model_output_dir)
tokenizer.save_pretrained(model_output_dir) 