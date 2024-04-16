"""
This module fine-tunes a GatorTron model in a regression way to make AUDIT-C scoring.
It uses the dataset as the `run_gen.py` (which uses a T5 and treat the task as a
virtually a single-token classification task). It has different input formats compared
to the T5 scripts.
"""

import os

import evaluate
import torch
import torch.nn as nn
import wandb
from datasets import load_from_disk, load_metric
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoConfig, AutoModel,
                          AutoModelForSequenceClassification, AutoTokenizer,
                          BertPreTrainedModel, EarlyStoppingCallback, Trainer,
                          TrainingArguments)

from utils import (DATASET_PATH, PROJECT_NAME, convert_to_dataframe,
                   expand_comorbidity)

__author__ = "hw56@indiana.edu"
__version__ = "0.0.1"
__license__ = "0BSD"


os.environ["TOKENIZERS_PARALLELISM"] = "false"
MODEL_NAME = "UFNLP/gatortron-base"
GATROTRON_MAX_LEN = 192
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class GatorTron_Dataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.df = df.reset_index(drop=True)  # drop index
        self.max_len = max_len
        self.texts = [
            ", ".join(f"{k}: {v}" for k, v in row.items() if k != "audit.c.score")
            for index, row in self.df.iterrows()
        ]
        self.labels = df["audit.c.score"].tolist()

        self.encodings = tokenizer(
            self.texts, truncation=True, padding="max_length", max_length=max_len
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float32)
        return item


class GatorTron_Regresser(BertPreTrainedModel):
    def __init__(self, model_name):
        config = AutoConfig.from_pretrained(model_name)
        super().__init__(config)
        self.gatortron = AutoModel.from_pretrained(model_name)
        self.cls_layer1 = nn.Linear(config.hidden_size, 128)
        self.relu1 = nn.ReLU()
        # self.ff1 = nn.Linear(128, 128)
        # self.tanh1 = nn.Tanh()
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)  # fixme: add dropout
        self.ff2 = nn.Linear(128, 1)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.gatortron(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.last_hidden_state[:, 0, :]
        output = self.cls_layer1(logits)
        output = self.relu1(output)
        # output = self.ff1(output)
        # output = self.tanh1(output)
        # output = self.dropout(output)  # fixme: apply dropout
        output = self.ff2(output).squeeze(-1)

        loss = None
        if labels is not None:
            loss_fn = nn.MSELoss()
            loss = loss_fn(output, labels.float())

        return {"loss": loss, "logits": output}


# tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# metrics for regression
mse_metric = evaluate.load("mse")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    mse = mse_metric.compute(predictions=predictions.squeeze(), references=labels)
    return {"mse": mse}


# load dataset
dataset = load_from_disk(DATASET_PATH)

train_df = expand_comorbidity(convert_to_dataframe(dataset["train"]))
val_df = expand_comorbidity(convert_to_dataframe(dataset["val"]))
test_df = expand_comorbidity(convert_to_dataframe(dataset["test"]))

train_dataset = GatorTron_Dataset(train_df, tokenizer, GATROTRON_MAX_LEN)
eval_dataset = GatorTron_Dataset(val_df, tokenizer, GATROTRON_MAX_LEN)
test_dataset = GatorTron_Dataset(test_df, tokenizer, GATROTRON_MAX_LEN)


# init model
config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=1)  # for regression
model = GatorTron_Regresser(MODEL_NAME).to(device)
run_name = "gatrotron_rgr"

training_args = TrainingArguments(
    output_dir=f"ckpts/{run_name}",
    overwrite_output_dir=False,
    num_train_epochs=10.0,
    do_train=True,
    do_eval=True,
    do_predict=True,
    evaluation_strategy="steps",
    auto_find_batch_size=True,
    gradient_accumulation_steps=4,
    learning_rate=3e-4,
    weight_decay=1e-1,
    logging_steps=50,
    eval_steps=100,
    # bf16=True,
    report_to="wandb",
    load_best_model_at_end=True,
    save_steps=100,
    save_total_limit=3,
    remove_unused_columns=True,
)
wandb.init(project=PROJECT_NAME, name=run_name, config=training_args)


# initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=20)],
)

# train the model
trainer.train()
