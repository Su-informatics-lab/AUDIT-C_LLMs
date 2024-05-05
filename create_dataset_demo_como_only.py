'''
Create AUDIT-C Scoring dataset for a quick and dirty pilot run using
`AUD_LLM_CX_04052024.csv`.

20,364 samples are for training, 2,000 for val, and 5,000 for test.
'''

import pandas as pd
from utils import CSV_DEMO_COMO_ONLY_PATH, DEMO_COMO_PARQUET_PATH, SEED
from sklearn.model_selection import train_test_split


demo_como = pd.read_csv(CSV_DEMO_COMO_ONLY_PATH)
# data split
remaining_data, validation_data = train_test_split(
    demo_como, test_size=2000, random_state=SEED
)
remaining_data, test_data = train_test_split(
    remaining_data, test_size=5000, random_state=SEED
)

validation_data["split"] = "validation"
test_data["split"] = "test"
remaining_data["split"] = "train"
final_df_split = pd.concat([remaining_data, validation_data, test_data])

# save parquet
final_df_split.to_parquet(DEMO_COMO_PARQUET_PATH, index=False)
