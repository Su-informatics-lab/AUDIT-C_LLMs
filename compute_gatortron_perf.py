import torch
import json
import numpy as np
from transformers import AutoTokenizer
from datasets import load_from_disk
from torch.utils.data import DataLoader
from lifelines.utils import concordance_index
from tqdm import tqdm

from utils import convert_to_dataframe, expand_comorbidity, DATASET_PATH
from run_gatortron_rgr import GatorTron_Regresser, GatorTron_Dataset, GATROTRON_MAX_LEN

# Configuration
MODEL_NAME = "UFNLP/gatortron-base"
CHECKPOINT_DIR = "ckpts/gatrotron_rgr"
CHECKPOINTS = ["checkpoint-11300", "checkpoint-11400", "checkpoint-9400"]
RESULTS_FILE = "evaluation_results.json"
BATCH_SIZE = 8  # Set based on your device capabilities

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load the dataset
dataset = load_from_disk(DATASET_PATH)
test_df = expand_comorbidity(convert_to_dataframe(dataset["test"]))
test_dataset = GatorTron_Dataset(test_df, tokenizer, GATROTRON_MAX_LEN)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

def evaluate_model(model, dataloader):
    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", unit="batch"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            logits = outputs['logits'].detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()

            predictions.extend(logits)
            actuals.extend(labels)

    predictions = np.array(predictions)
    actuals = np.array(actuals)
    mse = np.mean((predictions - actuals) ** 2)
    rmse = np.sqrt(mse)
    c_index = concordance_index(actuals, predictions)

    return mse, rmse, c_index

results = {}

# Evaluate each checkpoint
for checkpoint in CHECKPOINTS:
    model_path = f"{CHECKPOINT_DIR}/{checkpoint}"
    print(f"Evaluating {model_path}...")

    # Load the model from checkpoint
    model = GatorTron_Regresser(MODEL_NAME).to(device)
    model.load_state_dict(torch.load(f"{model_path}/pytorch_model.bin", map_location=device))

    # Compute metrics
    mse, rmse, c_index = evaluate_model(model, test_loader)

    results[checkpoint] = {
        "MSE": mse,
        "RMSE": rmse,
        "C-index": c_index
    }

    print(f"Checkpoint: {checkpoint}")
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    print(f"C-index: {c_index}")
    print("\n" + "="*50 + "\n")

# Save results to JSON file
with open(RESULTS_FILE, 'w') as f:
    json.dump(results, f, indent=4)

print(f"Results saved to {RESULTS_FILE}")
