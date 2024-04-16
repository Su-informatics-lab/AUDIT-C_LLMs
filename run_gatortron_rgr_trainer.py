import os

import torch
import torch.nn as nn
from datasets import load_from_disk, load_metric
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoConfig, AutoModel, AutoTokenizer,
                          AutoModelForSequenceClassification,
                          BertPreTrainedModel, Trainer, TrainingArguments)

from utils import DATASET_PATH, convert_to_dataframe, expand_comorbidity

os.environ["TOKENIZERS_PARALLELISM"] = "false"
MODEL_NAME = 'UFNLP/gatortron-base'
GATROTRON_MAX_LEN = 160
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class GatorTron_Dataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.df = df.reset_index(drop=True)
        self.max_len = max_len
        self.texts = [
            ' '.join(f'{k}: {v}' for k, v in row.items() if k != 'audit.c.score') for
            index, row in self.df.iterrows()]
        self.labels = df['audit.c.score'].tolist()

        self.encodings = tokenizer(self.texts, truncation=True, padding='max_length',
                                   max_length=max_len)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item


class GatorTron_Regresser(BertPreTrainedModel):
    def __init__(self, model_name):
        # initialize config
        config = AutoConfig.from_pretrained(model_name)
        super().__init__(config)

        self.gatortron = AutoModel.from_pretrained(model_name)

        # define the regression head layers
        self.cls_layer1 = nn.Linear(config.hidden_size, 128)  # 1024, 128
        self.relu1 = nn.ReLU()
        # self.ff1 = nn.Linear(128, 128)
        # self.tanh1 = nn.Tanh()
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)  # fixme: add dropout
        self.ff2 = nn.Linear(128, 1)

        # initialize weights for newly defined layers
        self.init_weights()

    def forward(self, input_ids, attention_mask):
        # Feed the input to the GatorTron model to obtain contextualized representations
        outputs = self.gatortron(input_ids=input_ids, attention_mask=attention_mask)

        # get the representations of [CLS]
        logits = outputs.last_hidden_state[:, 0, :]

        # pass through the regression head
        output = self.cls_layer1(logits)
        output = self.relu1(output)
        # output = self.ff1(output)
        # output = self.tanh1(output)
        # output = self.dropout(output)  # fixme: apply dropout
        output = self.ff2(output)
        output = output.squeeze(-1)

        return output


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load dataset
dataset = load_from_disk(DATASET_PATH)

train_df = expand_comorbidity(convert_to_dataframe(dataset["train"]))
val_df = expand_comorbidity(convert_to_dataframe(dataset["val"]))
# test_df = expand_comorbidity(convert_to_dataframe(dataset["test"]))

train_dataset = GatorTron_Dataset(train_df, tokenizer, GATROTRON_MAX_LEN)
eval_dataset = GatorTron_Dataset(val_df, tokenizer, GATROTRON_MAX_LEN)
# test_dataset = GatorTron_Dataset(test_df, tokenizer, GATROTRON_MAX_LEN)


# init model
config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=1)  # regression
model = AutoModelForSequenceClassification.from_pretrained(GatorTron_Regresser(MODEL_NAME),
                                                           config=config).to(device)


# model = GatorTron_Regresser(MODEL_NAME).to("cuda")


# training_args = TrainingArguments(
#     output_dir=f"ckpts/{run_name}",
#     overwrite_output_dir=False,
#     num_train_epochs=50.0,
#     do_train=True,
#     do_eval=True,
#     do_predict=True,
#     evaluation_strategy="steps",
#     auto_find_batch_size=True,
#     gradient_accumulation_steps=4,
#     learning_rate=1e-5,
#     weight_decay=1e-1,
#     logging_steps=50,
#     eval_steps=100,
#     # bf16=True,
#     report_to="wandb",
#     load_best_model_at_end=True,
#     save_steps=100,
#     save_total_limit=3,
#     remove_unused_columns=True,
# )
# wandb.init(project=PROJECT_NAME, name=run_name, config=training_args)



# Define Training Arguments
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # number of training epochs
    per_device_train_batch_size=8,   # batch size for training
    per_device_eval_batch_size=8,    # batch size for evaluation
    # warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False
)

# initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=lambda p: load_metric("mse").compute(predictions=p.predictions,
                                                         references=p.label_ids),
)

# train the model
trainer.train()
