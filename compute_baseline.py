"""
Calculate random forest and linear regression baselines for AUDIT-C Scoring task.
"""

import numpy as np
import pandas as pd
from datasets import load_from_disk
from lifelines.utils import concordance_index
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from utils import DATASET_PATH, SEED

dataset = load_from_disk(DATASET_PATH)
# define preprocessing for categorical features (one-hot encoding)
categorical_transformer = OneHotEncoder(handle_unknown="ignore")


__author__ = "hw56@indiana.edu"
__version__ = "0.0.1"
__license__ = "0BSD"


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
        pd.DataFrame: DataFrame with expanded comorbidity features.
    """
    # split the comorbidity strings into lists
    comorbidity_lists = df[comorbidity_col].str.split(separator)

    # get the set of unique comorbidities
    unique_comorbidities = set()
    for com_list in comorbidity_lists.dropna():
        unique_comorbidities.update(com_list)

    # initialize columns for each comorbidity
    for comorbidity in unique_comorbidities:
        df[f"comorbidity_{comorbidity}"] = comorbidity_lists.apply(
            lambda x: int(comorbidity in x) if isinstance(x, list) else 0
        )

    return df.drop(
        comorbidity_col, axis=1
    )  # optionally drop the original comorbidity column


train_df = convert_to_dataframe(dataset["train"])
test_df = convert_to_dataframe(dataset["test"])
train_df_expanded = expand_comorbidity(train_df)
test_df_expanded = expand_comorbidity(test_df)


# separate features and target variable
X_train = train_df_expanded.drop("audit.c.score", axis=1)
y_train = train_df_expanded["audit.c.score"]
X_test = test_df_expanded.drop("audit.c.score", axis=1)
y_test = test_df_expanded["audit.c.score"]

# feature engineering
categorical_features = ["gender", "race", "ethnicity"]

preprocessor = ColumnTransformer(
    transformers=[("cat", categorical_transformer, categorical_features)],
    remainder="passthrough",
)

print("*" * 90)
print("Calculating random forest and linear regression baselines:")

###
# random forest
rf_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(random_state=SEED + 534)),
    ]
)

rf_pipeline.fit(X_train, y_train)

rf_y_pred = rf_pipeline.predict(X_test)

rf_mse = mean_squared_error(y_test, rf_y_pred)
rf_c_index = concordance_index(y_test, rf_y_pred)

print("Random Forest Baseline:\n")
print(f"\tMSE: {rf_mse} (RMSE: {np.sqrt(rf_mse)})")
print(f"\t(C-index): {rf_c_index}\n")

###
# linear regression
lr_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ]
)

lr_pipeline.fit(X_train, y_train)

lr_y_pred = lr_pipeline.predict(X_test)

lr_mse = mean_squared_error(y_test, lr_y_pred)
lr_c_index = concordance_index(y_test, lr_y_pred)

print("Linear Regression Baseline:\n")
print(f"\tMSE: {lr_mse} (RMSE: {np.sqrt(lr_mse)})")
print(f"\t(C-index): {lr_c_index}\n")

print("*" * 90)
