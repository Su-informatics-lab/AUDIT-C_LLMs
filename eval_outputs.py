import numpy as np
import torch
from datasets import load_from_disk
from sklearn.metrics import mean_squared_error
from transformers import AutoTokenizer, T5ForConditionalGeneration

from run_gen import HEAD, MAX_LENGTH, TAIL
from utils import DATASET_PATH, MAX_OUTPUT_LENGTH, SEED

# val_loss=9.049*e-9 (lr=3e-5, grad_accumu=4, auto_bs)
BEST_FLANT5_CKPT = "ckpts/sft_generation_flan-t5-base_lr3e-5/checkpoint-2400"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def output2score(prediction):

    score = int(prediction.strip())

    assert 0 <= prediction <= 12

    return score


def generate_prediction(model, tokenizer, input_text):
    """
    Generate prediction using the trained model with greedy decoding.

    Args:
        model: The trained T5 model.
        tokenizer: Tokenizer for the T5 model.
        input_text (str): The input text properly formatted for the model.

    Returns:
        str: The model's predicted output as a string.
    """
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    generated_ids = model.generate(input_ids, max_length=MAX_OUTPUT_LENGTH)
    prediction = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    return prediction


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


if __name__ == "__main__":

    torch.manual_seed(SEED + 2177)

    test_split = load_from_disk(DATASET_PATH)["test"]
    # fixme
    model = T5ForConditionalGeneration.from_pretrained(BEST_FLANT5_CKPT).to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        BEST_FLANT5_CKPT, max_length=MAX_LENGTH, padding_side="right", truncation=True
    )
    predicted_scores = []
    true_scores = []

    # Assuming you have a 'test_dataset' loaded similarly to 'dataset["val"]'
    for example in test_split:
        body = (
            f"Gender={example['gender']},\nRace={example['race']},"
            f"\nEthnicity={example['ethnicity']},\nAge={example['age']},"
            f"\nComorbidity={example['comorbidity']}\n"
        )
        prediction = generate_prediction(model, tokenizer, HEAD + body + TAIL + " ")
        score = output2score(prediction)
        predicted_scores.append(score)
        true_scores.append(example["audit.c.score"])

    # evaluate MSE and Acc
    mse = evaluate_mse(true_scores, predicted_scores)
    accuracy = evaluate_accuracy(true_scores, predicted_scores)

    print(f"MSE: {mse}")
    print(f"Accuracy: {accuracy}")
