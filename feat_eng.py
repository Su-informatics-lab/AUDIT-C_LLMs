"""
# AUDIT-C Scoring with Demographics, Comorbidities, and Drugs

This module contains scripts to draw a volcano plot (as a positive control) to determine
 if each individual feature is informative for the decision-making process of the
 AUDIT-C score. This is followed by a comparison of vanilla baselines with more informed
 baselines.
"""

import os
import pandas as pd
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from random import sample
from itertools import chain
from multiprocessing import Pool, cpu_count
from statsmodels.stats.multitest import multipletests
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from lifelines.utils import concordance_index
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from copy import deepcopy
from sklearn.svm import SVR

# pd.set_option('display.max_rows', 50)

# constants
PERSON_DRUG_HALFYEAR_PATH = 'gs://fc-secure-19ab668e-266f-4a5f-9c63-febea17b23cf/data/AUD_LLM_6MonthDrug_CX_06272024.csv'
DEMO_COMO_THREE_DRUG_PARQUET_PATH = 'gs://fc-secure-19ab668e-266f-4a5f-9c63-febea17b23cf/data/hw56/AUD_LLM_DEMO_COMO_THREE_DRUG_05032024.parquet'
DEMO_COMO_6MON_DRUG_PARQUET_PATH = 'gs://fc-secure-19ab668e-266f-4a5f-9c63-febea17b23cf/data/hw56/AUD_LLM_DEMO_COMO_6MON_DRUG_06302024.parquet'
VOLCANO_DEMO_COMO_6MON_DRUG_PARQUET_PATH = 'gs://fc-secure-19ab668e-266f-4a5f-9c63-febea17b23cf/data/hw56/AUD_LLM_VOLCANO_DEMO_COMO_6MON_DRUG_06302024.parquet'
MIN_PVALUE = 1e-100
ALPHA = 0.05  # collective risk we accept within a group (we got three groups: demo, como, drug)

###
# 0. Utilities
###
# we first need to match up df's data with that found in df_person_drug
# the key is 'person_id' (or 'PERSON_ID' in df_person_drug)
# reserve df's 'person_id', 'gender', 'race', 'ethnicity', 'age', 'comorbidity',
# 'q1.score', 'q2.score', 'q3.score', 'audit.c.score'
# and bring in df_person_drug's 'STANDARD_CONCEPT_NAME'
def match_and_merge_data(df, df_person_drug):
    """
    This function matches up df's data with that found in df_person_drug,
    reserving specific columns from df and bringing in the corresponding
    data from df_person_drug based on 'person_id'.

    Parameters:
        df: DataFrame containing the main data.
        df_person_drug: DataFrame containing drug data used in recent a half year.

    Returns:
        Merged DataFrame with the reserved columns from df and the
               'standard_concept_name' column from df_person_drug.
    """
    df_filtered = df[df['person_id'].isin(df_person_drug['PERSON_ID'])]

    # reserve columns from df
    reserved_columns = ['person_id', 'gender', 'race', 'ethnicity', 'age', 'comorbidity',
                        'q1.score', 'q2.score', 'q3.score', 'audit.c.score']
    df_reserve = df_filtered[reserved_columns]

    # merge the reserved columns with the 'STANDARD_CONCEPT_NAME' from df_person_drug
    merged_df = pd.merge(df_reserve, df_person_drug[['PERSON_ID', 'STANDARD_CONCEPT_NAME']],
                         how='left', left_on='person_id', right_on='PERSON_ID')

    # drop the redundant and lower case column name
    merged_df.drop(columns=['PERSON_ID'], inplace=True)
    merged_df.rename(columns={'STANDARD_CONCEPT_NAME': 'standard_concept_name'}, inplace=True)

    return merged_df


def create_binary_column(drug, df):
    return drug, df.apply(
        lambda row: int(drug in row['standard_concept_name'].split(' | ')), axis=1
    )


def create_binary_column_wrapper(args):
    return create_binary_column(*args)


def expand_drugs_to_binary_columns(df_matched):
    # get unique drugs
    unique_drugs = list(set(chain(
        *(r.split(' | ') for r in df_matched.standard_concept_name.tolist()))))

    # use multiprocessing to create binary columns in parallel
    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap(create_binary_column_wrapper,
                                      [(drug, df_matched) for drug in unique_drugs]),
                            total=len(unique_drugs),
                            desc="Processing drugs"))

    # convert the results to a DataFrame
    binary_drug_columns = {drug: binary_column for drug, binary_column in results}
    binary_drug_df = pd.DataFrame(binary_drug_columns)

    # concatenate the binary columns with the original DataFrame
    df_matched = pd.concat([df_matched, binary_drug_df], axis=1)

    return df_matched


# constants
MIN_PVALUE = 1e-100
ALPHA = 0.05  # collective risk we accept within a group (we got three groups: demo, como, drug)


def encode_categorical_with_reference(df, column, reference):
    # referenece: man/white/non-hispanic
    categories = [reference] + [x for x in df[column].unique() if x != reference]
    return pd.Categorical(df[column], categories=categories).codes


def calculate_fold_change_and_pval(feature, data_subset, is_categorical):
    if is_categorical:
        # get log2 fold change for categorical features
        pos_group = data_subset[data_subset['group'] == 1][feature]
        neg_group = data_subset[data_subset['group'] == 0][feature]

        mean_pos = np.mean(pos_group)
        mean_neg = np.mean(neg_group)

        fc = mean_pos / (mean_neg + 1e-12)
        log2fc = np.log2(fc + 1e-12)

        # calculate p-value using Mann-Whitney U Test
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html
        # The Mann-Whitney U test is a nonparametric test of the null hypothesis that the distribution underlying
        # sample x is the same as the distribution underlying sample y.
        _, pval = stats.mannwhitneyu(pos_group, neg_group, alternative='two-sided')
    else:
        # calculate p-value using t-test for continuous features (age)
        pos_group = data_subset[data_subset['group'] == 1][feature]
        neg_group = data_subset[data_subset['group'] == 0][feature]
        log2fc = np.log2(pos_group.mean() / (neg_group.mean() + 1e-12))
        _, pval = stats.ttest_ind(pos_group, neg_group, alternative='two-sided')

    return feature, log2fc, pval


def worker(data):
    feature, df_encoded, is_categorical = data
    return calculate_fold_change_and_pval(feature, df_encoded, is_categorical)


def prepare_volcano_data(df, unique_drugs, unique_comorbidities, ignore_columns=[]):
    df_encoded = df.copy()
    df_encoded['gender'] = encode_categorical_with_reference(df_encoded, 'gender',
                                                             'Man')
    df_encoded['race'] = encode_categorical_with_reference(df_encoded, 'race', 'White')
    df_encoded['ethnicity'] = encode_categorical_with_reference(df_encoded, 'ethnicity',
                                                                'Others')

    # add a group column indicating the label
    df_encoded['group'] = np.where(
        ((df_encoded['gender'] == 0) & (df_encoded['audit.c.score'] >= 4)) |
        ((df_encoded['gender'] == 1) & (df_encoded['audit.c.score'] >= 3)),
        'pos', 'neg'
    )
    # positive group is coded as 1 (aka, neg is the reference group)
    df_encoded['group'] = encode_categorical_with_reference(df_encoded, 'group', 'neg')

    # identify categorical and continuous features
    categorical_features = ['gender', 'race', 'ethnicity'] + list(
        unique_comorbidities) + list(unique_drugs)
    continuous_features = ['age']

    # prepare feature list with their type, excluding ignored columns
    features = [(f, df_encoded[[f, 'group']], f in categorical_features) for f in
                df_encoded.columns if
                f not in ['audit.c.score', 'group'] + ignore_columns]

    # calculate fold change and p-values
    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap(worker, features), total=len(features),
                            desc="Calculating fold changes and p-values"))

    # split results into groups
    demo_features = ['gender', 'race', 'ethnicity', 'age']
    drug_features = unique_drugs
    como_features = unique_comorbidities

    demo_results = [r for r in results if r[0] in demo_features]
    drug_results = [r for r in results if r[0] in drug_features]
    como_results = [r for r in results if r[0] in como_features]

    # apply FDR correction within each group
    def apply_fdr(results):
        pvals = [r[2] for r in results]
        _, pvals_corrected, _, _ = multipletests(pvals,
                                                 method='fdr_bh')  # Benjamini/Hochberg
        for i, result in enumerate(results):
            results[i] = (result[0], result[1], result[2], pvals_corrected[i])
        return results

    demo_results = apply_fdr(demo_results)
    drug_results = apply_fdr(drug_results)
    como_results = apply_fdr(como_results)

    # prepare df for volcano
    volcano_data = {
        'Feature': [r[0] for r in demo_results + drug_results + como_results],
        'Log2FC': [r[1] for r in demo_results + drug_results + como_results],
        'P-value': [r[2] for r in demo_results + drug_results + como_results],
        'P-value Corrected': [r[3] for r in demo_results + drug_results + como_results]
    }

    volcano_df = pd.DataFrame(volcano_data)
    volcano_df['-Log10P Corrected'] = -np.log10(volcano_df['P-value Corrected'] + 1e-12)

    return volcano_df, demo_results, drug_results, como_results, features, df_encoded


def count_significant_features(volcano_df, alpha=ALPHA):
    # count the number of features with adjusted p-values less than the threshold
    significant_count = (volcano_df['P-value Corrected'] < alpha).sum()
    return significant_count


def plot_volcano(volcano_df, title, save_to=None):
    # create volcano plot
    plt.figure(figsize=(12, 8))
    plt.scatter(volcano_df['Log2FC'], volcano_df['-Log10P Corrected'], color='blue')
    plt.axhline(y=-np.log10(ALPHA), color='r',
                linestyle='--')  # Standard significance threshold
    plt.axhline(y=-np.log10(volcano_df['P-value Corrected']).min(), color='green',
                linestyle='--', label='FDR threshold')  # FDR threshold
    plt.ylim(-0.01, 5)
    plt.xlim(-5, 5)

    plt.xlabel('Log2 Fold Change')
    plt.ylabel('-Log10 of Corrected P-value')
    plt.title(title)
    plt.legend()

    if save_to:
        plt.savefig(save_to)
    else:
        plt.show()

# if __name__ == '__main__':

###
# 1. Prepare Data
###
df = pd.read_parquet(DEMO_COMO_THREE_DRUG_PARQUET_PATH)
df_person_drug = pd.read_csv(PERSON_DRUG_HALFYEAR_PATH)
print(f'{df.shape=} with columns {df.columns}', '\n\n')
print(f'{df_person_drug.shape=} with columns {df_person_drug.columns}')
df_matched = match_and_merge_data(df, df_person_drug)


# expand drugs (15,344 of them) into binary coded columns
print('num unque drugs:', f"{len(set(chain(*[r.split(' | ') for r in df_matched.standard_concept_name.tolist()])))}")
# unique_drugs is useful in following computation, should be calculated anyway
unique_drugs = set(chain(*[r.split(' | ') for r in df_matched.standard_concept_name.tolist()]))
assert len(unique_drugs) == 15344

# expand period separated comorbidity (18 of them) into binary coded columns
# unique_comorbidities is useful in following computation, should be calculated anyway
unique_comorbidities = set()
for comorbidities in df_matched['comorbidity']:
    if comorbidities is not None:
        unique_comorbidities.update(comorbidities.split(','))
assert len(unique_comorbidities) == 18

# encode only once
if not os.path.exists(DEMO_COMO_6MON_DRUG_PARQUET_PATH):
    # encode comorbidity
    # init df_matched with binary comorbidities as 0
    for comorbidity in unique_comorbidities:
        df_matched[comorbidity] = 0
    # populate it
    for index, row in tqdm(df_matched.iterrows()):
        if row['comorbidity'] is not None:
            for comorbidity in row['comorbidity'].split(','):
                df_matched.at[index, comorbidity] = 1
    # encode drug
    df_matched = expand_drugs_to_binary_columns(df_matched)
else:
    pd.read_parquet(DEMO_COMO_6MON_DRUG_PARQUET_PATH)

###
# 2. Volcano Plot

# The volcano plot customarily has an x-axis with log2 fold change and a y-axis with -log10 p-value.
# Specifically, we:
# - Divide a population of 180k patients into positive (with AUDIT-C scores: men >= 4 and women >= 3) and negative groups.
# - Compute the log2 fold change by comparing the two groups, positive and negative.
# - Compute the -log10 p-value (Benjamini-Hochberg adjusted) using p-values from the Mann-Whitney U Test (for categorical variables) and Pearson correlation (for continuous variables).
###

# remember to drop columns not needed
rm_columns = ['q1.score', 'q2.score', 'q3.score', 'person_id', 'comorbidity']
df_matched = df_matched.drop(columns=rm_columns)

volcano_df, _, _, _, _, _ = prepare_volcano_data(df_matched,
                                                 unique_drugs,
                                                 unique_comorbidities,
                                                 ignore_columns=['split'])
volcano_df.to_parquet(VOLCANO_DEMO_COMO_6MON_DRUG_PARQUET_PATH)

plot_volcano(volcano_df, f'Volcano Plot for AUDIT-C Score',
             save_to='figure/volcano_demo_como_6mon_drug.jpg')
