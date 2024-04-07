import json
import os
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from sklearn.metrics import mean_squared_error
from tqdm.auto import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration

from utils import DATASET_PATH, HEAD, MAX_LENGTH, SEED, TAIL

# fixme
# val_loss=9.049*e-9 (lr=3e-5, grad_accumu=4, auto_bs, outputlen=4)
BEST_FLANT5_CKPT = "ckpts/sft_generation_flan-t5-base/checkpoint-2400"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
result_dir = Path("results")
result_dir.mkdir(exist_ok=True)
result_filename = "_".join(BEST_FLANT5_CKPT.split("/")[-2:]) + ".json"


def output2score(prediction):

    # only reserve the token for score after TAIL (i.e., "\n### AUDIT-C Score:")
    score = int(prediction.split((TAIL).strip())[-1].strip())

    assert 0 <= score <= 12

    return score


#
# def generate_prediction(model, tokenizer, input_text):
#     """
#     Generate prediction using the trained model with greedy decoding.
#
#     Args:
#         model: The trained T5 model.
#         tokenizer: Tokenizer for the T5 model.
#         input_text (str): The input text properly formatted for the model.
#
#     Returns:
#         str: The model's predicted output as a string.
#     """
#     input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
#     generated_ids = model.generate(input_ids, max_length=MAX_LENGTH)
#     prediction = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
#
#     return prediction


def evaluate_mse(true_scores, predicted_scores):
    """
    Compute the Mean Squared Error (MSE) between true scores and predicted scores.

    Args:
        true_scores (list of int): The ground truth scores.
        predicted_scores (list of int): The scores predicted by the model.

    Returns:
        float: The computed MSE.
    """

    return mean_squared_error(true_scores, predicted_scores)


def evaluate_accuracy(true_scores, predicted_scores):
    """
    Compute the accuracy as the proportion of exact matches between true scores and
    predicted scores.

    Args:
        true_scores (list of int): The ground truth scores.
        predicted_scores (list of int): The scores predicted by the model.

    Returns:
        float: The accuracy as a proportion of exact matches.
    """
    true_scores = np.array(true_scores)
    predicted_scores = np.array(predicted_scores)
    accuracy = np.mean(true_scores == predicted_scores)

    return accuracy


def batch_generate_predictions(model, tokenizer, input_texts, batch_size=8):
    """
    Generate predictions for a batch of input texts with a progress bar.

    Args:
        model: The trained T5 model.
        tokenizer: Tokenizer for the T5 model.
        input_texts (list of str): The list of input texts properly formatted for the
            model.
        batch_size (int): Batch size for processing.

    Returns:
        List[str]: The list of model's predicted outputs as strings.
    """
    model.eval()  # Ensure model is in evaluation mode
    predictions = []

    # Wrap the range function with tqdm for a progress bar
    for i in tqdm(range(0, len(input_texts), batch_size),
                  desc="Generating predictions"):
        batch = input_texts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True,
                           max_length=MAX_LENGTH).to(device)

        with torch.no_grad():  # No need to track gradients
            outputs = model.generate(**inputs, max_length=MAX_LENGTH)
        batch_predictions = [tokenizer.decode(output, skip_special_tokens=True) for
                             output in outputs]
        predictions.extend(batch_predictions)

    return predictions


if __name__ == "__main__":

    torch.manual_seed(SEED + 2177)
    test_split = load_from_disk(DATASET_PATH)["test"]
    # fixme
    model = T5ForConditionalGeneration.from_pretrained(BEST_FLANT5_CKPT).to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        BEST_FLANT5_CKPT, max_length=MAX_LENGTH, padding_side="right", truncation=True
    )

    # Prepare the input texts for batch processing
    input_texts = [
        HEAD
        + (
            f"Gender={example['gender']},\nRace={example['race']},"
            f"\nEthnicity={example['ethnicity']},\nAge={example['age']},"
            f"\nComorbidity={example['comorbidity']}\n"
        )
        + TAIL
        + " "
        for example in test_split
    ]

    # Generate predictions in batches
    batch_predictions = batch_generate_predictions(
        model, tokenizer, input_texts, batch_size=8
    )

    # Convert predictions to scores
    predicted_scores = [output2score(pred) for pred in batch_predictions]
    true_scores = [example["audit.c.score"] for example in test_split]

    # Evaluate MSE and Accuracy
    mse = evaluate_mse(true_scores, predicted_scores)
    accuracy = evaluate_accuracy(true_scores, predicted_scores)
    print(f"MSE: {mse}")
    print(f"Accuracy: {accuracy}")

    # saving
    results_data = {
        "mse": mse,
        "accuracy": accuracy,
        "predicted_scores": predicted_scores,
        "true_scores": true_scores,
    }
    results_path = os.path.join(result_dir, result_filename)
    with open(results_path, "w") as json_file:
        json.dump(results_data, json_file, indent=4)

    print(f"Results saved to {results_path}")
