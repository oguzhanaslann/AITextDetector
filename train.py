from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import load_from_disk
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType


label2id = {"HUMAN": 0, "AI": 1}
id2label = {0: "HUMAN", 1: "AI"}

def map_labels_of_dataframe(frame):
  frame["label"] = frame["label"].map(label2id)
  return frame

model_name = "roberta-base"
tokenizer = RobertaTokenizerFast.from_pretrained(model_name) # Python-based

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_training_dataset = load_from_disk("data/tokenized_training")
tokenized_validation_dataset = load_from_disk("data/tokenized_validation")

model = RobertaForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(label2id),
    label2id=label2id,
    id2label=id2label
)


output_dir = "./results"

training_args = TrainingArguments(
    output_dir=output_dir,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir=f'{output_dir}/logs',
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    report_to="none" # Add this line to disable wandb logging
)


peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
)

peft_model = get_peft_model(model, peft_config)

peft_model.print_trainable_parameters()


peft_lora_finetuning_trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_training_dataset,
    eval_dataset=tokenized_validation_dataset
)   

peft_lora_finetuning_trainer.train()


model_output_dir = "./finetuned_roberta"
peft_model.save_pretrained(model_output_dir)
tokenizer.save_pretrained(model_output_dir)
