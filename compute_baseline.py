"""
Calculate random forest and linear regression baselines for AUDIT-C Scoring task.
"""

import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

from utils import CSV_DEMO_COMO_ONLY_PATH, SEED, expand_comorbidity

# df = load_from_disk(CSV_DEMO_COMO_ONLY_PATH)
# define preprocessing for categorical features (one-hot encoding)
categorical_transformer = OneHotEncoder(handle_unknown="ignore")


__author__ = "hw56@indiana.edu"
__version__ = "0.0.2"
__license__ = "0BSD"

df = pd.read_csv(CSV_DEMO_COMO_ONLY_PATH)
train_df, test_df = train_test_split(test_size=5000, random_state=SEED)

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

print("*" * 80)
print("Calculating baselines for AUDIT-C Scoring:")

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
print(f"\tC-index: {rf_c_index}\n")
print("*" * 40)

###
# linear regression
lr_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("scaler", StandardScaler()),
        ("regressor", LinearRegression())
    ]
)

lr_pipeline.fit(X_train, y_train)

lr_y_pred = lr_pipeline.predict(X_test)

lr_mse = mean_squared_error(y_test, lr_y_pred)
lr_c_index = concordance_index(y_test, lr_y_pred)

print("Linear Regression Baseline:\n")
print(f"\tMSE: {lr_mse} (RMSE: {np.sqrt(lr_mse)})")
print(f"\tC-index: {lr_c_index}\n")
print("*" * 40)

# Ridge Regression
ridge_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("scaler", StandardScaler()),
        ("regressor", Ridge(alhpa=100.0))  # best alpha in grid search over e-2 to e4
    ]
)

ridge_pipeline.fit(X_train, y_train)

ridge_y_pred = ridge_pipeline.predict(X_test)

ridge_mse = mean_squared_error(y_test, ridge_y_pred)
ridge_c_index = concordance_index(y_test, ridge_y_pred)

print("Ridge Regression Baseline:\n")
print(f"\tMSE: {ridge_mse} (RMSE: {np.sqrt(ridge_mse)})")
print(f"\tC-index: {ridge_c_index}\n")
print("*" * 40)


# Elastic Net Regression
en_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("scaler", StandardScaler()),
        ("regressor", ElasticNet())
    ]
)

en_pipeline.fit(X_train, y_train)

en_y_pred = en_pipeline.predict(X_test)

en_mse = mean_squared_error(y_test, en_y_pred)
en_c_index = concordance_index(y_test, en_y_pred)

print("Elastic Net Regression Baseline:\n")
print(f"\tMSE: {en_mse} (RMSE: {np.sqrt(en_mse)})")
print(f"\tC-index: {en_c_index}\n")
print("*" * 40)

###
# Support Vector Machine (SVM)
svm_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("scaler", MinMaxScaler()),  # standardizing
        ("regressor", SVR(kernel='linear')),
    ]
)

svm_pipeline.fit(X_train, y_train)

svm_y_pred = svm_pipeline.predict(X_test)

svm_mse = mean_squared_error(y_test, svm_y_pred)
svm_c_index = concordance_index(y_test, svm_y_pred)

print("Support Vector Machine (SVM) Baseline:\n")
print(f"\tMSE: {svm_mse} (RMSE: {np.sqrt(svm_mse)})")
print(f"\tC-index: {svm_c_index}\n")
print("*" * 80)
