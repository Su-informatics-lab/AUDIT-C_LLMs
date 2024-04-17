import torch
from transformers import AutoConfig, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_from_disk
from lifelines.utils import concordance_index

from utils import convert_to_dataframe, expand_comorbidity, DATASET_PATH
from run_gatortron_rgr import GatorTron_Regresser, GatorTron_Dataset, GATROTRON_MAX_LEN

# configuration
MODEL_NAME = "UFNLP/gatortron-base"
CHECKPOINT_DIR = "ckpts/gatrotron_rgr"
CHECKPOINTS = ["checkpoint-11300", "checkpoint-11400", "checkpoint-9400"]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# load the dataset
dataset = load_from_disk(DATASET_PATH)
test_df = expand_comorbidity(convert_to_dataframe(dataset["test"]))
test_dataset = GatorTron_Dataset(test_df, tokenizer, GATROTRON_MAX_LEN)


def compute_metrics(predictions, labels):
    mse = ((predictions - labels) ** 2).mean()
    rmse = mse ** 0.5
    c_index = concordance_index(labels, predictions)
    return mse, rmse, c_index


# evaluate each checkpoint
for checkpoint in CHECKPOINTS:
    model_path = f"{CHECKPOINT_DIR}/{checkpoint}"
    print(f"Evaluating {model_path}...")

    # Load the model from checkpoint
    model = GatorTron_Regresser(MODEL_NAME).to(device)
    model.load_state_dict(torch.load(f"{model_path}/pytorch_model.bin", map_location=device))

    # Prepare the trainer
    training_args = TrainingArguments(
        output_dir=".",
        do_predict=True,
        per_device_eval_batch_size=8
    )

    trainer = Trainer(
        model=model,
        args=training_args
    )

    # Make predictions
    predictions = trainer.predict(test_dataset)
    mse, rmse, c_index = compute_metrics(predictions.predictions.squeeze(), test_df['audit.c.score'].values)

    print(f"Checkpoint: {checkpoint}")
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    print(f"C-index: {c_index}")
    print("\n" + "="*50 + "\n")
