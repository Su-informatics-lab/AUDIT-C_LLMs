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

"""

__author__ = "hw56@indiana.edu"
__version__ = "0.0.1"
__license__ = "0BSD"

import os
from typing import List

import torch
import wandb
from datasets import DatasetDict, load_from_disk
from transformers import (T5ForConditionalGeneration, AutoTokenizer,
                          EarlyStoppingCallback, TrainingArguments)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

from utils import DATASET_PATH, MODEL_NAME, PROJECT_NAME, SEED

os.environ["TOKENIZERS_PARALLELISM"] = "false"
run_name = f'sft_generation_{MODEL_NAME.split("/")[-1]}'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="right")
HEAD = ("### Score the user's Alcohol Use Disorders Identification Test (AUDIT-C) "
        "from 0 to 12 based on the provided demographics and comorbidity data:\n")
TAIL = "### AUDIT-C Score:\n"

collator = DataCollatorForCompletionOnlyLM(TAIL, tokenizer=tokenizer)


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
        score = example["audit.c.score"][i]
        output_texts.append(HEAD + body + TAIL + score)

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
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=4,
        learning_rate=3e-4,
        weight_decay=1e-1,
        logging_steps=100,
        eval_steps=1000,
        # bf16=True,
        report_to="wandb",
        load_best_model_at_end=True,
        save_steps=1000,
        save_total_limit=3,
        remove_unused_columns=True,
    )
    wandb.init(project=PROJECT_NAME, name=run_name, config=training_args)

    trainer = SFTTrainer(
        model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"].select(range(320)),  # fixme: can be a bad choice
        formatting_func=formatting_func,
        data_collator=collator,
        # fixme
        max_seq_length=1,  # a score
        args=training_args,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()
