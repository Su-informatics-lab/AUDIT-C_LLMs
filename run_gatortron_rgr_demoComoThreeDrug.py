"""
This module fine-tunes a GatorTron model in a regression way to make AUDIT-C scoring.
It uses the dataset as the `run_gen.py` (which uses a T5 and treat the task as a
virtually a single-token classification task). It has different input formats compared
to the T5 scripts.

This model makes use of demographics, comorbidity, and drug to make predictions of
AUDIT-C scores.
"""

import argparse
import os

import pandas as pd
import torch
import torch.nn as nn
import wandb
from torch.utils.data import Dataset
from transformers import (AutoConfig, AutoModel, AutoTokenizer,
                          BertPreTrainedModel, EarlyStoppingCallback, Trainer,
                          TrainingArguments)

from utils import (DEMO_COMO_THREE_DRUG_PARQUET_PATH, PROJECT_NAME, SEED,
                   compute_metrics, expand_comorbidity,
                   period_separated_column_concatenation_formatting)

__author__ = "hw56@indiana.edu"
__version__ = "0.0.2"
__license__ = "0BSD"


# os.environ["TOKENIZERS_PARALLELISM"] = "false"
MODEL_NAME = "UFNLP/gatortron-base"
# max input len 467
GATROTRON_MAX_LEN = 512 - 32
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class GatorTron_Dataset(Dataset):
    def __init__(self, df, tokenizer, max_len, with_date):
        self.df = df.reset_index(drop=True)  # drop index
        self.max_len = max_len
        self.texts = [
            period_separated_column_concatenation_formatting(row, with_date=with_date)
            for _, row in self.df.iterrows()
        ]
        self.labels = [int(label) for label in self.df["audit.c.score"].tolist()]

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
    def __init__(self, model_name, non_linear_head=False):
        config = AutoConfig.from_pretrained(model_name)
        super().__init__(config)
        self.non_linear_head = non_linear_head
        self.gatortron = AutoModel.from_pretrained(model_name)
        self.cls_layer1 = nn.Linear(config.hidden_size, 128)
        self.relu1 = nn.ReLU()
        if self.non_linear_head:
            self.ff1 = nn.Linear(128, 128)
            self.tanh1 = nn.Tanh()
            # dropout rate = 0.1, for GatorTron-base
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.ff2 = nn.Linear(128, 1)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.gatortron(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.last_hidden_state[:, 0, :]
        output = self.cls_layer1(logits)
        output = self.relu1(output)
        if self.non_linear_head:
            output = self.ff1(output)
            output = self.tanh1(output)
            output = self.dropout(output)
        output = self.ff2(output).squeeze(-1)

        loss = None
        if labels is not None:
            loss_fn = nn.MSELoss()
            loss = loss_fn(output, labels.float())

        return {"loss": loss, "logits": output}


if __name__ == "__main__":
    torch.manual_seed(SEED + 123565)
    parser = argparse.ArgumentParser(
        description="Train a GatorTron regression model for AUDIT-C Scoring."
    )
    parser.add_argument(
        "--non_linear_head",
        action='store_true',
        default=False,
        help="whether to use non-linear regression head"
    )
    parser.add_argument(
        "--with_date",
        action='store_true',
        default=False,
        help="whether to use date information"
    )
    parser.add_argument(
        "--model_name",
        default=MODEL_NAME,
        help="which model to use"
    )
    parser.add_argument(
        "--run_name",
        help="like 'gatrotron_rgr_demo_como_threeDrug_linear_head'"
    )
    args = parser.parse_args()

    # load dataset and tokenizer
    df = pd.read_parquet(DEMO_COMO_THREE_DRUG_PARQUET_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 1. expand comorbidity into multiple binary ratings for all possible comorbidity
    # 2. ignore 'q1.score', 'q2.score', 'q3.score', 'audit.c.score', 'person_id', and
    # 'split'
    # 3. regularize 'concept_name_1(_2/_3)' so that the corresponding concept_name and
    # drug_exposure_start_date stay together
    # e.g., Concept Name: [mask] (Exposure Start Date: [mask]), if with_date
    # else Concept Name: [mask]
    train_df = expand_comorbidity(df.loc[df.split == 'train'])
    val_df = expand_comorbidity(df.loc[df.split == 'validation'])
    test_df = expand_comorbidity(df.loc[df.split == 'test'])

    train_dataset = GatorTron_Dataset(train_df, tokenizer,
                                      GATROTRON_MAX_LEN, with_date=args.with_date)
    eval_dataset = GatorTron_Dataset(val_df, tokenizer,
                                     GATROTRON_MAX_LEN, with_date=args.with_date)
    test_dataset = GatorTron_Dataset(test_df, tokenizer,
                                     GATROTRON_MAX_LEN, with_date=args.with_date)

    # init model
    config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=1)  # for regression
    model = GatorTron_Regresser(MODEL_NAME, args.non_linear_head).to(device)

    training_args = TrainingArguments(
        output_dir=os.path.join("ckpts", args.run_name),
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
    wandb.init(project=PROJECT_NAME, name=args.run_name, config=training_args)

    # initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=50)],
    )

    # train the model
    trainer.train()

    # evaluate on the test set after training
    test_results = trainer.evaluate(test_dataset)
    wandb.log({"test_results": test_results})
    print("Test Results:", test_results)
