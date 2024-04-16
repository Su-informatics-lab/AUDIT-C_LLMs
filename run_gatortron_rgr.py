import os

import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_from_disk
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm, trange
from transformers import (AutoConfig, AutoModel, AutoTokenizer,
                          BertPreTrainedModel)

from utils import DATASET_PATH
from compute_baseline import expand_comorbidity, convert_to_dataframe

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class GatorTron_Dataset(Dataset):

    def __init__(self, df, tokenizer, max_len):
        self.df = df.reset_index()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):

        # demo and como of a patient
        datum = ', '.join(
            [f'{k}: {v}' for k, v in self.df.loc[index].items() if k != 'index'])

        tokens = self.tokenizer.tokenize(datum)
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        if len(tokens) < self.max_len:
            tokens = tokens + ['[PAD]' for _ in range(self.max_len - len(tokens))]
        else:
            tokens = tokens[:self.max_len - 1] + ['[SEP]']

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = torch.tensor(input_ids)

        attention_mask = (input_ids != 0).long()

        audit_c_score = torch.tensor(self.df.loc[index, 'audit.c.score'],
                                     dtype=torch.float16)

        return input_ids, attention_mask, audit_c_score


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

        return output


def train(model, criterion, optimizer, train_loader, val_loader, epochs, device,
          eval_interval):
    for epoch in trange(epochs, desc="epoch"):
        model.train()
        train_loss = 0
        for i, (input_ids, attention_mask, target) in enumerate(train_loader):
            optimizer.zero_grad()

            input_ids, attention_mask, target = input_ids.to(device), attention_mask.to(
                device), target.to(device)

            output = model(input_ids=input_ids, attention_mask=attention_mask)

            loss = criterion(output, target.type_as(output))
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Evaluate every 'eval_interval' steps
            if (i + 1) % eval_interval == 0:
                val_loss = evaluate(model=model, criterion=criterion,
                                    dataloader=val_loader, device=device)
                print(
                    f"Step {i + 1} of epoch {epoch}: Training loss:"
                    f" {train_loss / (i + 1)}, Validation Loss: {val_loss}")

        # end of epoch evaluation and logging
        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch} complete! Average Training Loss: {avg_train_loss}")
        val_loss = evaluate(model=model, criterion=criterion, dataloader=val_loader,
                            device=device)
        print(f"Epoch {epoch} complete! Validation Loss: {val_loss}")


def evaluate(model, criterion, dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for i, (input_ids, attention_mask, target) in enumerate(dataloader):
            input_ids, attention_mask, target = input_ids.to(device), attention_mask.to(
                device), target.to(device)
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(output, target.type_as(output))
            total_loss += loss.item()
    return total_loss / len(dataloader)


def predict(model, dataloader, device):
    predicted_label = []
    actual_label = []
    with torch.no_grad():
        for input_ids, attention_mask, target in (dataloader):
            input_ids, attention_mask, target = input_ids.to(device), attention_mask.to(
                device), target.to(device)
            output = model(input_ids, attention_mask)

            predicted_label += output
            actual_label += target

    return predicted_label


if __name__ == "__main__":

    MODEL_NAME = 'UFNLP/gatortron-base'
    LR = 3e-4
    MAX_LEN = 160
    BATCH_SIZE = 4
    NUM_THREADS = 4
    EVAL_INTERVAL = 20  # for test

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    criterion = nn.MSELoss()

    # load data
    dataset = load_from_disk(DATASET_PATH)

    train_df = expand_comorbidity(convert_to_dataframe(dataset["train"]))
    val_df = expand_comorbidity(convert_to_dataframe(dataset["val"]))
    test_df = expand_comorbidity(convert_to_dataframe(dataset["test"]))

    # load sft model
    model = GatorTron_Regresser(MODEL_NAME)
    model = model.to(device)
    # optimizer
    optimizer = optim.AdamW(params=model.parameters(), lr=LR)

    ## Training Dataset
    train_set = GatorTron_Dataset(train_df, tokenizer, MAX_LEN)
    valid_set = GatorTron_Dataset(val_df, tokenizer, MAX_LEN)
    test_set = GatorTron_Dataset(test_df, tokenizer, MAX_LEN)

    ## Data Loaders
    train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE,
                              num_workers=NUM_THREADS)
    valid_loader = DataLoader(dataset=valid_set, batch_size=BATCH_SIZE,
                              num_workers=NUM_THREADS)
    test_loader = DataLoader(dataset=test_set, batch_size=BATCH_SIZE,
                             num_workers=NUM_THREADS)

    print(len(train_loader))

    train(model=model,
          criterion=criterion,
          optimizer=optimizer,
          train_loader=train_loader,
          val_loader=valid_loader,
          epochs=3,
          device=device,
          eval_interval=EVAL_INTERVAL)
