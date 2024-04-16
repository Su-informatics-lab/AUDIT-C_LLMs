import pandas as pd

CSV_PATH = 'gs://fc-secure-19ab668e-266f-4a5f-9c63-febea17b23cf/data/hw56/AUD_LLM_CX_04052024.csv'
DATASET_PATH = 'gs://fc-secure-19ab668e-266f-4a5f-9c63-febea17b23cf/data/hw56/AUD_LLM_CX_04052024'
MODEL_NAME = 'google/flan-t5-base'
PROJECT_NAME = 'AUDIT-C_LLMs'
SEED = 6179

MAX_LENGTH = 256
MAX_OUTPUT_LENGTH = 4
HEAD = ("### Score the user's Alcohol Use Disorders Identification Test (AUDIT-C) "
        "from 0 to 12 based on the provided demographics and comorbidity data:\n")
TAIL = "\n### AUDIT-C Score:"



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