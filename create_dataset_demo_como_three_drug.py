'''
Create AUDIT-C Scoring dataset for a quick and dirty pilot run using
`AUD_LLM_Top3Drug_CX_05032024.csv`.

This dataset is a long-form

111,569 samples are for training, 5,000 for val, and 5,000 for test.
'''

from datasets import load_dataset, DatasetDict
import pandas as pd
from utils import (CSV_DEMO_COMO_ONLY_PATH,
                   CSV_THREE_DRUG_PATH,
                   DEMO_COMO_THREE_DRUG_PARQUET_PATH,
                   DEMO_COMO_THREE_DRUG_DATASET_PATH,
                   SEED)

demo_como = pd.read_csv(CSV_DEMO_COMO_ONLY_PATH)
drug = pd.read_csv(CSV_THREE_DRUG_PATH)
# merge the dataframes based on person_id
merged_df = pd.merge(demo_como, drug, on='person_id', how='left')

# create a sequential counter for each patient's drug exposure
merged_df['exposure_counter'] = merged_df.groupby('person_id').cumcount().add(1)

# pivot the long-form dataframe to wide-form
pivoted_df = merged_df.pivot_table(
    index=['person_id', 'gender', 'race', 'ethnicity', 'age', 'comorbidity',
           'q1.score', 'q2.score', 'q3.score', 'audit.c.score'],
    columns='exposure_counter',
    values=['concept_name', 'drug_exposure_start_date'],
    aggfunc='first')

# flatten the multi-index columns
pivoted_df.columns = [f'{col}_{num}' for col, num in pivoted_df.columns]
pivoted_df.reset_index(inplace=True)

# merge the pivoted dataframe with the original wide-form dataframe
final_df = pd.merge(demo_como, pivoted_df, on=['person_id', 'gender', 'race',
                                               'ethnicity', 'age', 'comorbidity',
                                               'q1.score', 'q2.score', 'q3.score',
                                               'audit.c.score'], how='left')
# filter out rows where the drug columns are null
final_df = final_df.dropna(subset=[col for col in final_df.columns if 'concept_name' in col])
# save parquet
final_df.to_parquet(DEMO_COMO_THREE_DRUG_PARQUET_PATH, index=False)

# read it back
dataset = load_dataset('parquet', data_files=DEMO_COMO_THREE_DRUG_PARQUET_PATH)
dataset = dataset.shuffle(seed=SEED)

train_test_split = dataset['train'].train_test_split(test_size=5000 + 2000, seed=SEED)
val_test_split = train_test_split['test'].train_test_split(test_size=5000, seed=SEED)

split_datasets = DatasetDict({
    'train': train_test_split['train'],  # 104,569
    'val': val_test_split['train'],  # 2,000
    'test': val_test_split['test']  # 5,000
})

split_datasets.save_to_disk(DEMO_COMO_THREE_DRUG_DATASET_PATH)
