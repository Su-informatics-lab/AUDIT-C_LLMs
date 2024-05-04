"""Create the AUDIT-C Scoring dataset for a quick and dirty pilot run using both
`AUD_LLM_CX_04052024.csv` (a wide-form table with demographics and comorbidity) and
`AUD_LLM_Top3Drug_CX_05032024.csv` (a long-form table that has person_id from the pool
of those in the wide-form who have at least one drug exposure record. Drug name
(`concept_name`) and administration timestamp (`drug_exposure_start_date`) are reported.

The resulting dataframe contains the following columns:
    - 'person_id', 'gender', 'race', 'ethnicity', 'age', 'comorbidity', 'q1.score',
        'q2.score', 'q3.score', 'audit.c.score' (from the wide-form)
    - 'concept_name_1', 'concept_name_2', 'concept_name_3',
        'drug_exposure_start_date_1', 'drug_exposure_start_date_2',
        'drug_exposure_start_date_3' (from the long-form)
    - 'split' (indicating data split: 173,666 for train, 2,000 for validation, and 5,000
        for test)

Note: The table has 65,837 null cells for comorbidity only.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from utils import (
    CSV_DEMO_COMO_ONLY_PATH,
    CSV_THREE_DRUG_PATH,
    DEMO_COMO_THREE_DRUG_PARQUET_PATH,
    SEED,
)

demo_como = pd.read_csv(CSV_DEMO_COMO_ONLY_PATH)
drug = pd.read_csv(CSV_THREE_DRUG_PATH)
# merge the dataframes based on person_id
merged_df = pd.merge(demo_como, drug, on="person_id", how="left")

# create a sequential counter for each patient's drug exposure
merged_df["exposure_counter"] = merged_df.groupby("person_id").cumcount().add(1)

# pivot the long-form dataframe to wide-form
pivoted_drug_df = merged_df.pivot_table(
    index="person_id",
    columns="exposure_counter",
    values=["concept_name", "drug_exposure_start_date"],
    aggfunc="first",
    fill_value="",
)

# flatten the multi-index columns
pivoted_drug_df.columns = [f"{col}_{num}" for col, num in pivoted_drug_df.columns]
pivoted_drug_df.reset_index(inplace=True)

# merge the pivoted dataframe with the original wide-form dataframe
# 180,666 (65837 null for comorbidity)
final_df = pd.merge(demo_como, pivoted_drug_df, on=["person_id"], how="inner")

# data split
remaining_data, validation_data = train_test_split(
    final_df, test_size=2000, random_state=SEED
)
remaining_data, test_data = train_test_split(
    remaining_data, test_size=5000, random_state=SEED
)
validation_data["split"] = "validation"
test_data["split"] = "test"
remaining_data["split"] = "train"
final_df_split = pd.concat([remaining_data, validation_data, test_data])

# save parquet
final_df_split.to_parquet(DEMO_COMO_THREE_DRUG_PARQUET_PATH, index=False)
