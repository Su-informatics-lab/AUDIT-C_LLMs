import pandas as pd
import numpy as np
from lifelines.utils import concordance_index
from sklearn.metrics import mean_squared_error

CSV_DEMO_COMO_ONLY_PATH = 'gs://fc-secure-19ab668e-266f-4a5f-9c63-febea17b23cf/data/hw56/AUD_LLM_CX_04052024.csv'
CSV_THREE_DRUG_PATH = 'gs://fc-secure-19ab668e-266f-4a5f-9c63-febea17b23cf/data/hw56/AUD_LLM_Top3Drug_CX_05032024.csv'
DEMO_COMO_DATASET_PATH = 'gs://fc-secure-19ab668e-266f-4a5f-9c63-febea17b23cf/data/hw56/AUD_LLM_CX_04052024'
DEMO_COMO_THREE_DRUG_PARQUET_PATH = 'gs://fc-secure-19ab668e-266f-4a5f-9c63-febea17b23cf/data/hw56/AUD_LLM_DEMO_COMO_THREE_DRUG_05032024.parquet'
DEMO_COMO_THREE_DRUG_DATASET_PATH = 'gs://fc-secure-19ab668e-266f-4a5f-9c63-febea17b23cf/data/hw56/AUD_LLM_DEMO_COMO_THREE_DRUG_05032024'
MODEL_NAME = 'google/flan-t5-base'
PROJECT_NAME = 'AUDIT-C_LLMs'
SEED = 6179

MAX_LENGTH = 256
MAX_OUTPUT_LENGTH = 4
HEAD = ("### Score the user's Alcohol Use Disorders Identification Test (AUDIT-C) "
        "from 0 to 12 based on the provided demographics and comorbidity data:\n")
TAIL = "\n### AUDIT-C Score:"


def period_separated_column_concatenation_formatting(a_row):
    return ", ".join(f"{k}: {v}" for k, v in a_row.items() if k != "audit.c.score")


def evaluate_mse(true_scores, predicted_scores):
    """
    Compute the Mean Squared Error (MSE) between true scores and predicted scores.

    Args:
        true_scores: The ground truth scores.
        predicted_scores: The scores predicted by the model.

    Returns:
        float: The computed MSE.
    """

    return mean_squared_error(true_scores, predicted_scores)


def evaluate_c_index(true_scores, predicted_scores):
    """
    Compute the concordance index (C-index).

    The C-index quantifies how well the model's predicted scores are able to rank
    samples in the same order as their true scores. A C-index of 0.5 suggests no better
    than random chance, and a C-index of 1.0 indicates perfect prediction performance.

    Args:
        true_scores: The ground truth scores.
        predicted_scores: The scores predicted by the model.

    Returns:
        float: The C-index as a measure of predictive accuracy.
    """

    return concordance_index(true_scores, predicted_scores)


def compute_metrics(eval_pred):
    """
    Compute eval metrics during training, required by Trainer.
    :param eval_pred:
    :return:
    """

    predictions, labels = eval_pred
    mse = evaluate_mse(labels, predictions.squeeze())
    c_index = evaluate_c_index(labels, predictions.squeeze())
    return {"mse": mse, "rmse": np.sqrt(mse), "c-index": c_index}


def convert_to_dataframe(dataset):
    df = pd.DataFrame(dataset)
    # select the relevant features and target variable
    df = df[["gender", "race", "ethnicity", "age", "comorbidity", "audit.c.score"]]
    return df


def expand_comorbidity(df, comorbidity_col="comorbidity", separator=","):
    """
    Expand the comorbidity column into multiple binary features.

    Args:
        df (pd.DataFrame): Input DataFrame.
        comorbidity_col (str): Name of the comorbidity column.
        separator (str): The separator used in the comorbidity strings.

    Returns:
        pd.DataFrame: DataFrame with expanded comorbidity features and the original
        comorbidity column removed.
    """
    # split the comorbidity strings into lists
    comorbidity_lists = df[comorbidity_col].str.split(separator)

    # get the set of unique comorbidities
    unique_comorbidities = set()
    for com_list in comorbidity_lists.dropna():
        unique_comorbidities.update(com_list)

    # initialize columns for each comorbidity
    for comorbidity in unique_comorbidities:
        # replace underscores with spaces
        comorbidity_cleaned = comorbidity.replace("_", " ")
        df[comorbidity_cleaned] = comorbidity_lists.apply(
            lambda x: int(comorbidity in x) if isinstance(x, list) else 0
        )

    df.drop(comorbidity_col, axis=1, inplace=True)

    return df