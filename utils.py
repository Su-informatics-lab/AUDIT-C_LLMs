import torch.nn as nn
import numpy as np
from lifelines.utils import concordance_index
from sklearn.metrics import mean_squared_error

CSV_DEMO_COMO_ONLY_PATH = 'gs://fc-secure-19ab668e-266f-4a5f-9c63-febea17b23cf/data/hw56/AUD_LLM_CX_04052024.csv'
CSV_THREE_DRUG_PATH = 'gs://fc-secure-19ab668e-266f-4a5f-9c63-febea17b23cf/data/hw56/AUD_LLM_Top3Drug_CX_05032024.csv'
DEMO_COMO_PARQUET_PATH = 'gs://fc-secure-19ab668e-266f-4a5f-9c63-febea17b23cf/data/hw56/AUD_LLM_CX_04052024.parquet'
DEMO_COMO_THREE_DRUG_PARQUET_PATH = 'gs://fc-secure-19ab668e-266f-4a5f-9c63-febea17b23cf/data/hw56/AUD_LLM_DEMO_COMO_THREE_DRUG_05032024.parquet'
DEMO_EXPCOMO_PIPE_SEP_HALFYEARDRUG_212K_RAW_PARQUET_PATH = 'gs://fc-secure-19ab668e-266f-4a5f-9c63-febea17b23cf/data/hw56/AUD_LLM_DEMO_ExpComo_PipeSep_HalfYearDrug_212K.parquet'
FATIGUE_ANXIETY = "gs://fc-secure-19ab668e-266f-4a5f-9c63-febea17b23cf/bq_exports/xiongc@researchallofus.org/20240801/survey_27943846/survey_27943846_*.csv"
DEMO_EXPCOMO_PIPE_SEP_HALFYEARDRUG_FAT_ANX_AUD_212K_RAW_PARQUET_PATH = 'gs://fc-secure-19ab668e-266f-4a5f-9c63-febea17b23cf/data/hw56/AUD_LLM_DEMO_ExpComo_PipeSep_HalfYearDrug_fatigue_anxiety_auditc_212K.parquet'
# the single source of truth
DEMO_EXPCOMO_PIPE_SEP_HALFYEARDRUG_FAT_ANX_AUD_DIABETE_INSURANCE_212K_RAW_PARQUET_PATH = 'gs://fc-secure-19ab668e-266f-4a5f-9c63-febea17b23cf/data/hw56/AUD_LLM_DEMO_ExpComo_PipeSep_HalfYearDrug_fatigue_anxiety_auditc_diabete_insurance_212K.parquet'
# the single source of truth w/ drug columns expanded
DEMO_EXPCOMO_EXPHALFYEARDRUG_FAT_ANX_AUD_DIABETE_INSURANCE_212K_RAW_PARQUET_PATH = 'gs://fc-secure-19ab668e-266f-4a5f-9c63-febea17b23cf/data/hw56/AUD_LLM_DEMO_ExpComo_ExpHalfYearDrug_fatigue_anxiety_auditc_diabete_insurance_212K.parquet'
# audit-c 212k data, w/ pos_es, neg_es, and standard_concept_name (pipe concatenated)
AUDIT_ML_PAQUET = 'gs://fc-secure-19ab668e-266f-4a5f-9c63-febea17b23cf/data/hw56/AUDIT_ML.parquet'

MODEL_NAME = 'google/flan-t5-base'
PROJECT_NAME = 'ALLOFUS'
SEED = 6179

MAX_LENGTH = 256
MAX_OUTPUT_LENGTH = 4
HEAD = ("### Score the user's Alcohol Use Disorders Identification Test (AUDIT-C) "
        "from 0 to 12 based on the provided demographics and comorbidity data:\n")
TAIL = "\n### AUDIT-C Score:"


def format_drug_column(df, prefix='Recent drug: ', suffix='.'):
    """
    Format a column of pipe-separated drug use names into a readable string
    format by replacing pipe separators with commas. Each formatted string starts
    with a specified `prefix` and ends with a specified `suffix`.

    """
    df = df.copy()
    df.loc[:, 'standard_concept_name'] = prefix + df['standard_concept_name'].str.replace(' | ', ', ') + suffix
    return df


def count_parameters(model: nn.Module) -> int:
    """
    Count the number of parameters in a model.
    Args:
        model (nn.Module): The model for which to count parameters.
    Returns:
        int: The total number of parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def period_separated_narrative_formatting(a_row, with_drug: bool = True) -> str:
    """
    Concatenates all the indicative columns into a single string for LLM consumption.

    Note:
        - Fields 'q1.score', 'q2.score', 'q3.score', 'audit.c.score', 'person_id',
            'anxiety', 'fatigue', and 'split' are ignored.
        - Numerated concept_names and drug_exposure_start_dates are renamed for
            stylistic consistency.
        - Most recent drug use is closer to comorbidity.

    An example is shown below.
        ```gender: Female; race: White; ethnicity: Non-Hispanic; age: 45;
        Rheumatic Disease; Diabetes wo C; Metastatic Solid Tumor; Liver Disease Moderate
        Severe; Recent drug: DrugA, DrugB, DrugC.```

    Args:
        a_row: A row in dataframe.
        with_drug: Whether to include drug information.
    Returns:
        A string of demographics, comorbidity, and drug use of a specific person.
    """
    demographics = []
    comorbidities = []
    drugs = []
    for k, v in a_row.items():
        if k in ['q1.score', 'q2.score', 'q3.score',
                 'anxiety', 'fatigue', 'empl_insurance_or_not',
                 'audit.c.score', 'person_id', 'split']:
            continue
        elif k in ['gender', 'race', 'ethnicity', 'age']:
            demographics.append(f"{k}: {v}")
        elif k == 'comorbidity':
            comorbidities.append(f'Comorbidity: {v}')
        elif k == 'standard_concept_name':
            if with_drug:
                drugs.append(f'Recent drug: {v}.')
        else:
            raise ValueError(f'Unknown key {k}')
    formatted_row = demographics + comorbidities + drugs
    return "; ".join(formatted_row)


def comorbidities_to_narrative(row, comorbidity_columns):
    """
    Converts comorbidity columns to narratives.
    """
    narratives = []
    for col in comorbidity_columns:
        if row[col] == 1:
            narratives.append(col.replace('_', ' '))
    if not narratives:
        return 'None'
    return ', '.join(narratives)


def period_separated_column_concatenation_formatting(a_row, with_date=True):
    """
    Concatenates all the indicative columns into a single string for LLM consumption.

    Note:
        - Fields 'q1.score', 'q2.score', 'q3.score', 'audit.c.score', 'person_id', and
        'split' are ignored.
        - Numerated concept_names and drug_exposure_start_dates are renamed for
            stylistic consistency.
        - Most recent drug use is closer to comorbidity.

    An example is shown below.
        ```gender: [mask], race: [mask], ethnicity: [mask], age: [mask],
        Rheumatic Disease: [mask], Diabetes wo C: [mask], Metastatic Solid Tumor:
        [mask], Liver Disease Moderate Severe: [mask], Myocardial Infarction: [mask],
        Peripheral Vascular Disease: [mask], Liver Disease Mild: [mask], Congestive
        Heart Failure: [mask], Chronic Pulmonary Disease: [mask], Dementia: [mask],
        Cerebrovascular Disease: [mask], Renal Disease Mild Moderate: [mask],
        Hemiplegia Paraplegia: [mask], Renal Disease Severe: [mask], Malignancy: [mask],
        Peptic Ulcer Disease: [mask], Diabetes w C: [mask], HIV: [mask],
        Concept Name: [mask] (Exposure Start Date: [mask]), Concept Name: [mask]
        (Exposure Start Date: [mask]), Concept Name: [mask] (Exposure Start Date:
        [mask])```

    Args:
        a_row: A row in dataframe.
        with_date: Whether to use date column.
    Returns:
        A string of demographics, comorbidity, and drug use of a specific person.
    """
    formatted_row = []
    drug_data_pairs = []
    for k, v in a_row.items():
        if k in ['q1.score', 'q2.score', 'q3.score',
                 'audit.c.score', 'person_id', 'split']:
            continue
        elif k.startswith('concept_name'):
            drug_number = k.split('_')[-1]
            drug_exposure_date = a_row[f'drug_exposure_start_date_{drug_number}']
            if with_date:
                drug_data_pairs.append(f'Concept Name: {v} (Exposure Start Date: {drug_exposure_date})')
            else:
                drug_data_pairs.append(f'Concept Name: {v}')
        elif k.startswith('drug_exposure_start_date'):
            continue
        else:
            formatted_row.append(f"{k}: {v}")
    formatted_row.extend(drug_data_pairs)
    return ", ".join(formatted_row)


# def period_separated_column_concatenation_formatting(a_row):
#     return ", ".join(f"{k}: {v}" for k, v in a_row.items() if k != "audit.c.score")


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


def compute_metrics_fine_grained(eval_pred):
    """
    Compute eval metrics during training, required by Trainer.
    :param eval_pred: A tuple containing predictions and labels.
    :return: A dictionary with computed metrics.
    """
    predictions, labels = eval_pred
    # Ensure predictions and labels are of the correct shape
    predictions = predictions.reshape(-1, 3)
    labels = labels.reshape(-1, 3)

    # Compute metrics for individual scores
    mse = evaluate_mse(labels, predictions)
    rmse = np.sqrt(mse)
    c_index = evaluate_c_index(labels.flatten(), predictions.flatten())

    # Compute overall AUDIT-C score metrics
    overall_pred = predictions.sum(axis=1)
    overall_true = labels.sum(axis=1)

    overall_mse = evaluate_mse(overall_true, overall_pred)
    overall_rmse = np.sqrt(overall_mse)
    overall_c_index = evaluate_c_index(overall_true, overall_pred)

    return {
        "individual_mse": mse,
        "individual_rmse": rmse,
        "individual_c-index": c_index,
        "mse": overall_mse,
        "rmse": overall_rmse,
        "c-index": overall_c_index
    }


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
    df = df.copy()
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
        df.loc[:, comorbidity_cleaned] = comorbidity_lists.apply(
            lambda x: int(comorbidity in x) if isinstance(x, list) else 0
        )
        # df[comorbidity_cleaned] = comorbidity_lists.apply(
        #     lambda x: int(comorbidity in x) if isinstance(x, list) else 0
        # )

    df.drop(comorbidity_col, axis=1, inplace=True)

    return df
