from datasets import load_dataset, load_metric
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import torch
import json

### MNLI dataset ###

dataset = load_dataset('glue', 'mnli')
metric = load_metric('glue', 'mnli')

# partial-input model

part_checkpoint = "bert-base-uncased"
batch_size = 8
part_tokenizer = AutoTokenizer.from_pretrained(part_checkpoint, use_fast=True)

part_model = AutoModelForSequenceClassification.from_pretrained(part_checkpoint, num_labels=3)

def part_preprocess_function(examples):
    return part_tokenizer(examples["hypothesis"], truncation=True)

part_dataset = dataset.map(part_preprocess_function, batched=True)
val_data = part_dataset['validation_matched']

part_args = TrainingArguments(
    "test-glue",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

part_trainer = Trainer(
    part_model,
    part_args,
    train_dataset=part_dataset["train"],
    eval_dataset=val_data,
    tokenizer=part_tokenizer,
    compute_metrics=compute_metrics
)

part_trainer.train()


# Main and tiny models (use both premise and hypothesis)

# Note: the reason of repeating some code blocks is that this code is unified from different jupyter notebooks

#model_checkpoint = "prajjwal1/bert-tiny" # for tiny
model_checkpoint = "bert-base-uncased" # for main
batch_size = 32 # clark2020: We found increasing the batch size did not change performance on the matched MNLI dev set

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=3)

def preprocess_function(examples):
    return tokenizer(examples["premise"], examples["hypothesis"], truncation=True)

encoded_dataset = dataset.map(preprocess_function, batched=True)
val_data = encoded_dataset['validation_matched']

args = TrainingArguments(
    "test-glue",
#   evaluation_strategy = "steps",
#   eval_steps=5000,
#   learning_rate=5e-5,
    logging_steps=7000,
    save_steps=7000,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
#   num_train_epochs=1,
#   weight_decay=0.01,
#   load_best_model_at_end=True,
#   metric_for_best_model="accuracy",
)

trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=val_data,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train() 
pred = trainer.predict(val_data)

### Fever dataset ###

# Following Mahabadi2020:
# train: "https://www.dropbox.com/s/v1a0depfg7jp90f/fever.train.jsonl"
# val: "https://www.dropbox.com/s/bdwf46sa2gcuf6j/fever.dev.jsonl"

label_dic = {'SUPPORTS':0, 'NOT ENOUGH INFO':1, 'REFUTES':2}

with open('./data/fever/fever.train.jsonl', 'r') as json_file:
    json_list = list(json_file)
fever_train_data = [json.loads(json_str) for json_str in json_list]
fever_train_evidence = [r['evidence'] for r in fever_train_data]
fever_train_claim = [r['claim'] for r in fever_train_data]
fever_train_label = [label_dic[r['gold_label']] for r in fever_train_data]

with open('./data/fever/fever.dev.jsonl', 'r') as json_file:
    json_list = list(json_file)
fever_val_data = [json.loads(json_str) for json_str in json_list]
fever_val_evidence = [r['evidence'] for r in fever_val_data]
fever_val_claim = [r['claim'] for r in fever_val_data]
fever_val_label = [label_dic[r['gold_label']] for r in fever_val_data]
with open('./data/fever/new_dev.jsonl', 'r') as json_file:
    json_list = list(json_file)
new_val_data = [json.loads(json_str) for json_str in json_list]
for i,l in [r['label'] for r in new_val_data]:
  if l == 'NOT ENOUGH INFO':
    fever_val_evidence.append(new_val_data[i]['evidence'])
    fever_val_claim.append(new_val_data[i]['claim'])
    fever_val_label.append(1)

model_checkpoint = "bert-base-uncased"
batch_size = 32
claim_only = True

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=3)

if claim_only:
    train_encodings = tokenizer(fever_train_claim, truncation=True, padding=True) 
    val_encodings = tokenizer(fever_val_claim, truncation=True, padding=True)
else:
    train_encodings = tokenizer(fever_train_evidence, fever_train_claim, truncation=True, padding=True)
    val_encodings = tokenizer(fever_val_evidence, fever_val_claim, truncation=True, padding=True)

class MakeDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
    
train_dataset = MakeDataset(train_encodings, fever_train_label)
val_dataset = MakeDataset(val_encodings, fever_val_label)

args = TrainingArguments(
    "fever-checkpoints",
    report_to = "none",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=3,
    weight_decay=0.1,
    warmup_steps=1000,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

trainer = Trainer(
    model,
    args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)
trainer.train()
pred = trainer.predict(val_dataset)
