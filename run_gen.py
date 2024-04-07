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

import torch
import wandb
from datasets import DatasetDict, load_from_disk
from transformers import (T5ForConditionalGeneration,
                          AutoTokenizer,
                          EarlyStoppingCallback,
                          TrainingArguments)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

from utils import DATASET_PATH, MODEL_NAME, PROJECT_NAME, SEED, MAX_OUTPUT_LENGTH

os.environ["TOKENIZERS_PARALLELISM"] = "false"
run_name = f'sft_generation_{MODEL_NAME.split("/")[-1]}'
# fixme: 256 should be enough but not optimal
MAX_LENGTH = 256
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME,
                                          max_length=MAX_LENGTH,
                                          padding_side="right",
                                          truncation=True)
HEAD = ("### Score the user's Alcohol Use Disorders Identification Test (AUDIT-C) "
        "from 0 to 12 based on the provided demographics and comorbidity data:\n")
TAIL = "\n### AUDIT-C Score:"

# collator = DataCollatorForCompletionOnlyLM(TAIL, tokenizer=tokenizer)


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


if __name__ == "__main__":
    torch.manual_seed(SEED + 21)

    dataset = load_from_disk(DATASET_PATH)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME,
                                                       # torch_dtype=torch.bfloat16
                                                       )

    training_args = TrainingArguments(
        output_dir=f"ckpts/{run_name}",
        overwrite_output_dir=False,
        num_train_epochs=50.0,
        do_train=True,
        do_eval=True,
        do_predict=True,
        evaluation_strategy="steps",
        auto_find_batch_size=True,
        # per_device_train_batch_size=16,
        # per_device_eval_batch_size=16,
        gradient_accumulation_steps=1,
        learning_rate=5e-6,
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

    trainer = SFTTrainer(
        model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
        formatting_func=formatting_func,
        # data_collator=collator,
        max_seq_length=MAX_LENGTH + MAX_OUTPUT_LENGTH,  # input_len + output_len
        args=training_args,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()
