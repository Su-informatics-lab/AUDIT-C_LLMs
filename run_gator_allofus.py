"""
This module fine-tunes a GatorTron model for regression to predict fatigue or anxiety
levels using a subpopulation of 212,000 individuals. It leverages demographics,
comorbidities, and recent drug use data.
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
                   DEMO_EXPCOMO_PIPE_SEP_HALFYEARDRUG_FAT_ANX_AUD_212K_RAW_PARQUET_PATH,
                   DEMO_EXPCOMO_PIPE_SEP_HALFYEARDRUG_FAT_ANX_AUD_DIABETE_INSURANCE_212K_RAW_PARQUET_PATH)

__author__ = "hw56@indiana.edu"
__version__ = "0.0.1"
__license__ = "0BSD"


# os.environ["TOKENIZERS_PARALLELISM"] = "false"
MODEL_NAME = "UFNLP/gatortron-base"
GATROTRON_MAX_LEN = 512
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class GatorTron_Dataset(Dataset):
    def __init__(self, df, target, with_drug, tokenizer, max_len=GATROTRON_MAX_LEN):
        self.df = df.reset_index(drop=True)  # drop index
        self.max_len = max_len
        self.texts = [
            period_separated_narrative_formatting(row, with_drug)
            for _, row in self.df.iterrows()
        ]
        self.labels = [int(label) for label in self.df[target].tolist()]

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


severity_mappings = {
    'fatigue': {
        'Average Fatigue 7 Days: None': 0,
        'Average Fatigue 7 Days: Mild': 1,
        'Average Fatigue 7 Days: Moderate': 2,
        'Average Fatigue 7 Days: Severe': 3,
        'Average Fatigue 7 Days: Very Severe': 4,
    },
    'anxiety': {
        'Emotional Problem 7 Days: Never': 0,
        'Emotional Problem 7 Days: Rarely': 1,
        'Emotional Problem 7 Days: Sometimes': 2,
        'Emotional Problem 7 Days: Often': 3,
        'Emotional Problem 7 Days: Always': 4,
    }
}


def encode_severity(df, column_name, mapping):
    """
    Encode severity levels in a DataFrame column using a specified mapping.

    Parameters:
    - df: pandas DataFrame
    - column_name: str, name of the column to encode
    - mapping: dict, mapping from severity levels to integers

    Returns:
    - pandas Series with encoded severity levels
    """
    return df[column_name].map(mapping)


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


if __name__ == "__main__":
    torch.manual_seed(SEED + 123565)
    parser = argparse.ArgumentParser(description="Train a GatorTron regression model for fatigue or anxiety prediction")
    parser.add_argument("--target", type=str, choices=['fatigue', 'anxiety', 'empl_insurance_or_not'])
    parser.add_argument("--with_drug", action='store_true', default=False,
                        help="if use drug data; set to False will only use demo + como")
    parser.add_argument("--non_linear_head", action='store_true', default=False, help="whether to use non-linear regression head")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--model_name", default=MODEL_NAME, help="which model to use")
    parser.add_argument("--run_name", help="like 'gatortron_demo_como_drug'")

    args = parser.parse_args()

    wandb.init(project=PROJECT_NAME, name=args.run_name)
    if args.target in ['fatigue', 'anxiety']:
        df = pd.read_parquet(
            DEMO_EXPCOMO_PIPE_SEP_HALFYEARDRUG_FAT_ANX_AUD_212K_RAW_PARQUET_PATH)
        # filter out nas
        df = df[df[args.target] != 'PMI: Skip']
        # encode target
        df[args.target] = encode_severity(df, args.target, severity_mappings[args.target])

    elif args.target == 'empl_insurance_or_not':
        df = pd.read_parquet(
            DEMO_EXPCOMO_PIPE_SEP_HALFYEARDRUG_FAT_ANX_AUD_DIABETE_INSURANCE_212K_RAW_PARQUET_PATH)
        # filter out individuals without any insurance
        df = df[(df['empl.merge'] == 1) | (df['non-empl.merge'] == 1)]
        # create a new column 'insurance_type'
        # 0 indicates employer-based and 1 indicates non-employer-based
        df['insurance_type'] = df['non-empl.merge'].astype(int)
        # drop the original 'empl.merge' and 'non-empl.merge' columns
        df = df.drop(columns=['empl.merge', 'non-empl.merge',
                              'q1.score', 'q2.score', 'q3.score', 'audit.c.score',
                              'fatigue', 'anxiety', 'type_2_diabetes', 'type_1_diabetes',
                              'other_unknown_diabetes', 'prediabetes'])
    else:
        raise ValueError(f"Unknown target {args.target}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_df = preprocess_df_for_lm(df.loc[df.split == 'train'])
    val_df = preprocess_df_for_lm(df.loc[df.split == 'validation'])
    test_df = preprocess_df_for_lm(df.loc[df.split == 'test'])

    train_dataset = GatorTron_Dataset(train_df, args.target, args.with_drug, tokenizer)
    eval_dataset = GatorTron_Dataset(val_df, args.target, args.with_drug, tokenizer)
    test_dataset = GatorTron_Dataset(test_df, args.target, args.with_drug, tokenizer)

    # init model
    config = AutoConfig.from_pretrained(args.model_name, num_labels=1)  # for regression
    model = GatorTron_Regresser(args.model_name, args.non_linear_head).to(device)

    training_args = TrainingArguments(
        output_dir=os.path.join("ckpts", args.run_name),
        overwrite_output_dir=False,
        num_train_epochs=50.0,
        do_train=True,
        do_eval=True,
        do_predict=True,
        eval_strategy="steps",
        auto_find_batch_size=True,
        gradient_accumulation_steps=4,
        learning_rate=args.learning_rate,
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
