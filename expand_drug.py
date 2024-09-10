"""
# Expended Single Source of Truth

To make our experiments more consistent and resilient across runs, we incorporate extra
labels for the 212k population by gradually appending new labels (i.e., fatigue and
audit.c.source) to a table.
We call this table the Single Source of Truth (SST). Currently, the SST is hosted at:
```
DEMO_EXPCOMO_PIPE_SEP_HALFYEARDRUG_FAT_ANX_AUD_DIABETE_INSURANCE_212K_RAW_PARQUET_PATH
= 'gs://fc-secure-19ab668e-266f-4a5f-9c63-febea17b23cf/data/hw56/AUD_LLM_DEMO_ExpComo_PipeSep_HalfYearDrug_fatigue_anxiety_auditc_diabete_insurance_212K.parquet'```

This module processes the SST to expand the drug column ("standard_concept_name")
beforehand for subsequent individual drug effect correlation with the labels. This is
because this operation often takes around 6 hours.
"""

from itertools import chain
from multiprocessing import Pool, cpu_count

import pandas as pd
from tqdm import tqdm
from utils import (
    DEMO_EXPCOMO_EXPHALFYEARDRUG_FAT_ANX_AUD_DIABETE_INSURANCE_212K_RAW_PARQUET_PATH,
    DEMO_EXPCOMO_PIPE_SEP_HALFYEARDRUG_FAT_ANX_AUD_DIABETE_INSURANCE_212K_RAW_PARQUET_PATH)


def create_binary_column(drug, df):
    return drug, df.apply(
        lambda row: int(drug in row['standard_concept_name'] if pd.notna(row['standard_concept_name']) else 0), axis=1
    )


def create_binary_column_wrapper(args):
    return create_binary_column(*args)


def expand_drugs_to_binary_columns(df, unique_drugs):
    """
    This function expands the 'standard_concept_name' column in df into multiple binary columns,
    where each column corresponds to a unique drug and indicates its presence (1) or absence (0) for each row.

    Parameters:
        df: DataFrame with a 'standard_concept_name' column.

    Returns:
        DataFrame with the original columns plus additional binary columns for each unique drug.
    """
    # use multiprocessing to create binary columns in parallel
    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap(create_binary_column_wrapper, [(drug, df) for drug in unique_drugs]),
                            total=len(unique_drugs),
                            desc="Processing drugs"))

    # convert the results to a DataFrame
    binary_drug_columns = {drug: binary_column for drug, binary_column in results}
    binary_drug_df = pd.DataFrame(binary_drug_columns)

    # concatenate the binary columns with the original DataFrame
    df = pd.concat([df, binary_drug_df], axis=1)

    return df


if __name__ == '__main__':
    df = pd.read_parquet(DEMO_EXPCOMO_PIPE_SEP_HALFYEARDRUG_FAT_ANX_AUD_DIABETE_INSURANCE_212K_RAW_PARQUET_PATH)
    print(f"{df.columns=}")

    # expand drugs (15,355 in total) into binary coded columns
    print('number of unque drugs:', f"{len(set(chain(*[r.split(' | ') for r in df.standard_concept_name.dropna().tolist()])))}")
    unique_drugs = set(chain(*[r.split(' | ') for r in df.standard_concept_name.dropna().tolist()]))
    assert len(unique_drugs) == 15355

    df_w_drug_cols_expanded = expand_drugs_to_binary_columns(df, unique_drugs)
    # save to the single source of truth w/ drug columns expanded
    df_w_drug_cols_expanded.to_parquet(DEMO_EXPCOMO_EXPHALFYEARDRUG_FAT_ANX_AUD_DIABETE_INSURANCE_212K_RAW_PARQUET_PATH)
