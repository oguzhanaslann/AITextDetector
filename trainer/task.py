import os
import argparse
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import load_from_disk
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, required=True)
    parser.add_argument('--train-data', type=str, required=True)
    parser.add_argument('--valid-data', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=8)
    return parser.parse_args()

def train_model(args):
    # Label configuration
    label2id = {"HUMAN": 0, "AI": 1}
    id2label = {0: "HUMAN", 1: "AI"}

    # Initialize model and tokenizer
    model_name = "roberta-base"
    tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label
    )

    # Load datasets
    tokenized_training_dataset = load_from_disk(args.train_data)
    tokenized_validation_dataset = load_from_disk(args.valid_data)

    # Configure LoRA
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1
    )
    peft_model = get_peft_model(model, peft_config)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.model_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        report_to="tensorboard"
    )

    # Initialize trainer
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=tokenized_training_dataset,
        eval_dataset=tokenized_validation_dataset
    )

    # Train and save
    trainer.train()
    peft_model.save_pretrained(args.model_dir)
    tokenizer.save_pretrained(args.model_dir)

if __name__ == '__main__':
    args = get_args()
    train_model(args) 