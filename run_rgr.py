"""
This module performs supervised fine-tuning on T5 variants using the AUDIT-C Scoring
dataset (`AUD_LLM_CX_04052024.csv`) as a **generation** task.
The dataset has 10 fields ('person_id', 'gender', 'race', 'ethnicity', 'age',
'comorbidity', 'q1.score', 'q2.score', 'q3.score', and 'audit.c.score'), and 212,364
rows.

We will be using six of the fields, specifically:
- Dependent vars:
    - 'gender',
    - 'race'
    - 'ethnicity'
    - 'age', and
     - 'comorbidity'

- Predicted var:
    - 'audit.c.score'

206,864 samples are for training, 500 for val, and 5,000 for test.
"""

__author__ = "hw56@indiana.edu"
__version__ = "0.0.1"
__license__ = "0BSD"

import os
from typing import List

import numpy as np
import torch
import wandb
from datasets import DatasetDict, load_from_disk
from sklearn.metrics import mean_squared_error
from torch import nn
from transformers import (AutoTokenizer, EarlyStoppingCallback,
                          PreTrainedModel, T5Config, T5EncoderModel,
                          Trainer, TrainingArguments)
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

from utils import DATASET_PATH, MODEL_NAME, PROJECT_NAME, SEED

os.environ["TOKENIZERS_PARALLELISM"] = "false"
run_name = f'sft_regression_{MODEL_NAME.split("/")[-1]}'
# fixme: 256 should be enough but not optimal
MAX_LENGTH = 256
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME,
                                          max_length=MAX_LENGTH,
                                          padding_side="right",
                                          truncation=True)
HEAD = ("### Score the user's Alcohol Use Disorders Identification Test (AUDIT-C) "
        "from 0 to 12 based on the provided demographics and comorbidity data:\n")
TAIL = "\n### AUDIT-C Score:"


class T5EncoderForRegression(PreTrainedModel):
    config_class = T5Config

    def __init__(self, config):
        super().__init__(config)
        self.t5_encoder = T5EncoderModel(config)
        # predict a single continuous value
        self.regression_head = nn.Linear(config.d_model, 1)

    def forward(self, input_ids, attention_mask=None):
        encoder_outputs = self.t5_encoder(input_ids=input_ids,
                                          attention_mask=attention_mask)

        # use the output of the first token to predict the continuous value
        sequence_output = encoder_outputs.last_hidden_state
        pooled_output = sequence_output[:, 0]  # shape: (batch_size, hidden_size)
        regression_value = self.regression_head(pooled_output)  # shape: (batch_size, 1)

        return regression_value


class RegressionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs[0]
        # Reshape labels and logits for MSE computation
        logits = logits.view(-1).float()  # Ensure float for regression
        labels = labels.view(-1).float()
        loss_fct = nn.MSELoss()
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

def formatting_func(example: DatasetDict) -> List[str]:
    """
    Formats input examples by concatenating the source text with the target text,
    using the task-specific prefix and response template.

    Args:
        example: A dataset dictionary containing 'source' and 'target' fields.

    Returns:
        A list of formatted strings ready for model training.
    """
    output_texts = []

    for i in range(len(example["audit.c.score"])):
        body = (f"Gender={example['gender'][i]},\nRace={example['race'][i]},"
                f"\nEthnicity={example['ethnicity'][i]},\nAge={example['age'][i]},"
                f"\nComorbidity={example['comorbidity'][i]}\n")
        score = str(example["audit.c.score"][i])
        output_texts.append(HEAD + body + TAIL + ' ' + score)

    return output_texts


# mse for regression
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions[:, 0]  # Get the raw regression values
    return {"mse": mean_squared_error(labels, predictions)}


if __name__ == "__main__":
    torch.manual_seed(SEED + 21)

    dataset = load_from_disk(DATASET_PATH)
    model = T5EncoderForRegression.from_pretrained(MODEL_NAME)

    # Corrected TrainingArguments without compute_metrics
    training_args = TrainingArguments(
        output_dir=f"ckpts/{run_name}",
        overwrite_output_dir=False,
        num_train_epochs=50.0,
        do_train=True,
        do_eval=True,
        do_predict=True,
        evaluation_strategy="steps",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=1e-5,
        weight_decay=1e-1,
        logging_steps=50,
        eval_steps=500,
        report_to="wandb",
        load_best_model_at_end=True,
        save_steps=500,
        save_total_limit=3,
        remove_unused_columns=True,
    )
    wandb.init(project=PROJECT_NAME, name=run_name, config=training_args)

    trainer = RegressionTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
        compute_metrics=compute_metrics,
    )

    trainer.train()
