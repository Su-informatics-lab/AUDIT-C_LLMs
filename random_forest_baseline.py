import numpy as np
import pandas as pd
from datasets import load_from_disk
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from utils import DATASET_PATH

dataset = load_from_disk(DATASET_PATH)
# define preprocessing for categorical features (one-hot encoding)
categorical_transformer = OneHotEncoder(handle_unknown="ignore")


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


rf_pipeline = Pipeline(
    steps=[("preprocessor", preprocessor), ("regressor", RandomForestRegressor())]
)

# fit the model
rf_pipeline.fit(X_train, y_train)

# predictions
y_pred = rf_pipeline.predict(X_test)
y_pred_rounded = np.round(y_pred)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred_rounded)

print(f"Random Forest MSE: {mse}")
print(f"Random Forest Accuracy: {accuracy}")
