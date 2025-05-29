from transformers import RobertaTokenizer, RobertaForSequenceClassification
import pandas as pd
import datasets

def map_labels_of_dataframe(frame, label2id):
  frame["label"] = frame["label"].map(label2id)
  return frame

def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

def tokenize_datasets_and_save_to_disk():
    label2id = {"HUMAN": 0, "AI": 1}
    validation_dataframe = pd.read_csv('data/validation_dataset.csv')
    validation_dataframe = map_labels_of_dataframe(validation_dataframe, label2id)
    print(validation_dataframe.head())
    print(validation_dataframe.shape)
    validation_dataset = datasets.Dataset.from_pandas(validation_dataframe)

    # Load training dataset
    training_dataframe = pd.read_csv('data/combined_dataset.csv')
    training_dataframe = map_labels_of_dataframe(training_dataframe, label2id)
    # Print first few rows
    print(training_dataframe.head())

    # Print shape of the dataset
    print(training_dataframe.shape)

    # Convert to Hugging Face Dataset
    training_dataset = datasets.Dataset.from_pandas(training_dataframe)

    model_name = "roberta-base"

    tokenizer = RobertaTokenizer.from_pretrained(model_name) # Python-based
    tokenized__training_datasets = training_dataset.map(tokenize_function, batched=True)
    tokenized__validation_datasets = validation_dataset.map(tokenize_function, batched=True)

    tokenized__training_datasets.save_to_disk("data/tokenized_training")
    tokenized__validation_datasets.save_to_disk("data/tokenized_validation")
