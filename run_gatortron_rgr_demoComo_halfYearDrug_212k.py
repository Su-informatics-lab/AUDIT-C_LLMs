"""
This module fine-tunes a GatorTron model in a regression way to make AUDIT-C scoring.
It uses the subpopulation extracted from 212k individuals.

This model makes use of demographics, comorbidity, and drug use in the past six months
to make predictions of AUDIT-C scores.
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

from utils import (comorbidities_to_narrative, PROJECT_NAME, SEED,
                   compute_metrics, period_separated_narrative_formatting,
                   DEMO_EXPCOMO_PIPE_SEP_HALFYEARDRUG_212K_RAW_PARQUET_PATH)

__author__ = "hw56@indiana.edu"
__version__ = "0.0.1"
__license__ = "0BSD"


# os.environ["TOKENIZERS_PARALLELISM"] = "false"
MODEL_NAME = "UFNLP/gatortron-base"
GATROTRON_MAX_LEN = 512
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class GatorTron_Dataset(Dataset):
    def __init__(self, df, tokenizer, with_drug, max_len=GATROTRON_MAX_LEN):
        self.df = df.reset_index(drop=True)  # drop index
        self.max_len = max_len
        self.texts = [
            period_separated_narrative_formatting(row, with_drug)
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
        "--remove_zero_score",
        action='store_true',
        default=False,
        help="whether to filter out those w/ zero AUDIT-C score"
    )
    parser.add_argument(
        "--model_name",
        default=MODEL_NAME,
        help="which model to use"
    )
    parser.add_argument(
        "--with_drug",
        action='store_true',
        default=False,
        help="whether to use date information"
    )
    parser.add_argument(
        "--run_name",
        help="like 'gatortron_rgr_demo_como_withDrug_halfYearDrug_linear_head'"
    )
    args = parser.parse_args()

    # load dataset and tokenizer
    df = pd.read_parquet(DEMO_EXPCOMO_PIPE_SEP_HALFYEARDRUG_212K_RAW_PARQUET_PATH)

    # filter_out_zero_audit_c_persons
    # with columns: ['person_id', 'gender', 'race', 'ethnicity', 'age', 'q1.score',
    #        'q2.score', 'q3.score', 'audit.c.score', 'split',
    #        'standard_concept_name', 'Renal_Disease_Severe',
    #        'Renal_Disease_Mild_Moderate', 'Myocardial_Infarction',
    #        'Peripheral_Vascular_Disease', 'Dementia', 'Hemiplegia_Paraplegia',
    #        'Cerebrovascular_Disease', 'Rheumatic_Disease',
    #        'Chronic_Pulmonary_Disease', 'Malignancy', 'Congestive_Heart_Failure',
    #        'HIV', 'Diabetes_w_C', 'Peptic_Ulcer_Disease', 'Metastatic_Solid_Tumor',
    #        'Diabetes_wo_C', 'Liver_Disease_Mild', 'Liver_Disease_Moderate_Severe']
    if args.remove_zero_score:
        df = df.loc[df['audit.c.score'] > 0]  # 170375 rows × 29 columns

    def preprocess_df_for_lm(df):
        # convert " | " concatenated drug names to more readable forms
        df = df.copy()
        df.loc[:, 'standard_concept_name'] = df['standard_concept_name'].str.replace(' | ', ', ')
        comorbidity_columns = [
            'Renal_Disease_Severe', 'Renal_Disease_Mild_Moderate', 'Myocardial_Infarction',
            'Peripheral_Vascular_Disease', 'Dementia', 'Hemiplegia_Paraplegia',
            'Cerebrovascular_Disease', 'Rheumatic_Disease', 'Chronic_Pulmonary_Disease',
            'Malignancy', 'Congestive_Heart_Failure', 'HIV', 'Diabetes_w_C',
            'Peptic_Ulcer_Disease', 'Metastatic_Solid_Tumor', 'Diabetes_wo_C',
            'Liver_Disease_Mild', 'Liver_Disease_Moderate_Severe'
        ]

        # create a new column with the narrative form
        df.loc[:, 'comorbidity'] = df.apply(comorbidities_to_narrative, axis=1,
                                                 args=(comorbidity_columns,))
        # drop the original comorbidity columns
        df.drop(columns=comorbidity_columns, inplace=True)
        return df

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_df = preprocess_df_for_lm(df.loc[df.split == 'train'])
    val_df = preprocess_df_for_lm(df.loc[df.split == 'validation'])
    test_df = preprocess_df_for_lm(df.loc[df.split == 'test'])

    train_dataset = GatorTron_Dataset(train_df, tokenizer, args.with_drug)
    eval_dataset = GatorTron_Dataset(val_df, tokenizer, args.with_drug)
    test_dataset = GatorTron_Dataset(test_df, tokenizer, args.with_drug)

    # init model
    config = AutoConfig.from_pretrained(args.model_name, num_labels=1)  # for regression
    model = GatorTron_Regresser(args.model_name, args.non_linear_head).to(device)

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
        callbacks=[EarlyStoppingCallback(early_stopping_patience=30)],
    )

    # train the model
    trainer.train()

    # evaluate on the test set after training
    test_results = trainer.evaluate(test_dataset)
    wandb.log({"test_results": test_results})
    print("Test Results:", test_results)
