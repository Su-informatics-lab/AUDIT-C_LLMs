import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
from lifelines.utils import concordance_index
from sklearn.metrics import mean_squared_error
from torch import optim
from torch.utils.data import DataLoader, RandomSampler, TensorDataset

from cattransformer import CatTransformer
from utils import PROJECT_NAME, SEED, compute_metrics

# for reproducibility
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def save_model(model, eval_loss, top_models, run_name):
    model_dir = f"ckpts/{run_name}"
    os.makedirs(model_dir, exist_ok=True)
    model_path = f"{model_dir}/model_{eval_loss:.4f}.pt"
    torch.save(model.state_dict(), model_path)
    top_models.append((eval_loss, model_path))
    top_models = sorted(top_models, key=lambda x: x[0])[:3]  # keep only top 3 models
    return top_models


def prepare_standard_concept_name(df,
                                  prefix="Drugs used in the past half year may or may not reflect one's drinking behavior: "):
    column_name = 'standard_concept_name'
    # replace " | " with ", "
    df[column_name] = df[column_name].str.replace(" | ", ", ", regex=False)

    # substitute NaN values
    df[column_name] = df[column_name].fillna("No drug used in the past half year.")

    # optionally prepend the column values with a specified prefix
    df[column_name] = prefix + df[column_name]

    return df


# encode three categorical fields: gender, ethnicity, and race
def encode_categorical_with_reference(df, column, reference):
    # referenece: man/white/non-hispanic
    categories = [reference] + [x for x in df[column].unique() if x != reference]
    return pd.Categorical(df[column], categories=categories).codes


if __name__ == "__main__":
    torch.manual_seed(SEED)
    parser = argparse.ArgumentParser(
        description="Train a CatTransformer for AUDIT-C Scoring."
    )
    parser.add_argument("--with_drug_string", action='store_true', default=False, help="if use drug data; set to False will only use demo + como (as a TabTransformer)")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_interval", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=1000, help='early stopping patience')
    args = parser.parse_args()

    # initialize wandb
    run_name = "CatTransformer_demoComo_drug" if args.with_drug_string else "CatTransformer_demoComo"
    try:
        wandb.init(project=PROJECT_NAME, name=run_name)
    except wandb.errors.CommError:
        print("WandB communication error, proceeding without logging.")

    ### prepare data
    # we got raw (i.e., unencoded) data here: 212k patients with:
    # demo: 4 columns, gender, race, age, ethnicity
    # expanded como: 18 columns
    # pipe-separated drug in one column: 15,355 unique concepts; if one uses no drug in the past year, then it is a "NaN"
    # and other columns: scores (q1.score, q2.score, q3.score, audit.c.score), split, and person_id
    DEMO_EXPCOMO_PIPE_SEP_HALFYEARDRUG_212K_RAW_PARQUET_PATH = 'gs://fc-secure-19ab668e-266f-4a5f-9c63-febea17b23cf/data/hw56/AUD_LLM_DEMO_ExpComo_PipeSep_HalfYearDrug_212K.parquet'
    df = pd.read_parquet(
        DEMO_EXPCOMO_PIPE_SEP_HALFYEARDRUG_212K_RAW_PARQUET_PATH)
    df = prepare_standard_concept_name(df)

    # carefully encode gender, race, and ethnicity
    df['gender'] = encode_categorical_with_reference(df, 'gender',
                                                             'Man')
    df['race'] = encode_categorical_with_reference(df, 'race', 'White')
    df['ethnicity'] = encode_categorical_with_reference(df, 'ethnicity',
                                                                'Others')
    # reserve data we used for this run
    categorical_features = [
        "Metastatic_Solid_Tumor",
        "Dementia",
        "Peripheral_Vascular_Disease",
        "Myocardial_Infarction",
        "Peptic_Ulcer_Disease",
        "Hemiplegia_Paraplegia",
        "Congestive_Heart_Failure",
        "Cerebrovascular_Disease",
        "Diabetes_w_C",
        "Liver_Disease_Moderate_Severe",
        "Rheumatic_Disease",
        "Renal_Disease_Mild_Moderate",
        "HIV",
        "Renal_Disease_Severe",
        "Chronic_Pulmonary_Disease",
        "Malignancy",
        "Liver_Disease_Mild",
        "Diabetes_wo_C",
        "gender",
        "race",
        "ethnicity",
    ]

    continuous_features = ["age"]
    pred_var = ["audit.c.score"]
    all_cols = categorical_features + continuous_features + pred_var

    if args.with_drug_string:
        all_cols += ['standard_concept_name']

    df_train = df.loc[df["split"] == "train"][all_cols]  # 20,536 individuals * 23 cols
    df_val = df.loc[df["split"] == "validation"][all_cols]  # 2,000 individuals * 23 cols
    df_test = df.loc[df["split"] == "test"][all_cols]  # 5,000 individuals * 23 cols

    # convert the train, validation, and test sets to tensors
    x_categ_train = torch.tensor(df_train[categorical_features].values,
                                 dtype=torch.long)
    x_cont_train = torch.tensor(df_train[continuous_features].values, dtype=torch.float)
    y_train = torch.tensor(df_train["audit.c.score"].values,
                           dtype=torch.float).unsqueeze(1)

    x_categ_val = torch.tensor(df_val[categorical_features].values, dtype=torch.long)
    x_cont_val = torch.tensor(df_val[continuous_features].values, dtype=torch.float)
    y_val = torch.tensor(df_val["audit.c.score"].values, dtype=torch.float).unsqueeze(1)

    x_categ_test = torch.tensor(df_test[categorical_features].values, dtype=torch.long)
    x_cont_test = torch.tensor(df_test[continuous_features].values, dtype=torch.float)
    y_test = torch.tensor(df_test["audit.c.score"].values, dtype=torch.float).unsqueeze(
        1)

    cont_mean_std = (
        torch.Tensor(
            [
                df_train[continuous_features].mean().values,
                df_train[continuous_features].std().values,
            ]
        )
        .reshape(-1, 2)
        .to(device)
    )

    # find out unique values in each column for categorical values
    categories = [len(df_train[c].unique()) for c in categorical_features]
    assert len(categories) == len(categorical_features)

    model = CatTransformer(
        categories=categories,  # a list containing the number of unique values (i.e., levels) within each easily encodable category (categories of low cardinality)
        num_high_card_categories=0 if not args.with_drug_string else 1,  # fall back to a TabTransformer if no high_card_categ is in place
        num_continuous=len(continuous_features),  # number of continuous variables
        dim=32,  # input dimension/embedding size, paper set at 32
        depth=6,  # number of stacking transformer blocks, paper recommended 6
        heads=8,  # number of attention heads, paper recommends 8
        dim_head=16,  # vector length for each attention head
        dim_out=1,  # output dimension, fixed to 1 for regression
        mlp_hidden_mults=(4, 2),  # defines number of hidden layers of final MLP and multiplier of (bottom to top) input_size of (dim * num_categories) + num_continuous + dim
        mlp_act=nn.SiLU(),  # activation function for MLP
        continuous_mean_std=cont_mean_std,  # precomputed mean/std for continuous variables
        transformer_dropout=0.1,  # dropout for attention and residual links
        use_shared_categ_embed=True,  # share a fixed-length embeddings indicating the levels from the same column
        shared_categ_dim_divisor=8,  # 1/8 of cat_embedding dims are shared in CatCTransformer
        lm_model_name='UFNLP/gatortron-base', # Hugging Face BERT variant model name, and we recommend `Su-informatics-lab/gatortron_base_rxnorm_babbage_v2`
        lm_max_length=512,  # max tokens for LM embedding computation
        embeddings_cache_path='.lm_embeddings.pkl'  # path to cache embeddings
    ).to(device)

    # training settings
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    num_epochs = args.num_epochs
    batch_size = args.batch_size

    train_dataset = TensorDataset(x_categ_train, x_cont_train, y_train)
    val_dataset = TensorDataset(x_categ_val, x_cont_val, y_val)
    test_dataset = TensorDataset(x_categ_test, x_cont_test, y_test)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=RandomSampler(train_dataset)
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # early stopping and model saving settings
    early_stopping_steps = args.patience
    early_stopping_counter = 0
    best_eval_loss = float("inf")
    top_models = []

    ### train and eval
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        total_steps = 0

        for step, (x_categ_batch, x_cont_batch, y_batch) in enumerate(train_loader):
            optimizer.zero_grad()
            pred_train = model(x_categ_batch, x_cont_batch)
            loss = criterion(pred_train, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            total_steps += 1

            if total_steps % args.eval_interval == 0:
                avg_loss = running_loss / args.eval_interval
                c_index = concordance_index(
                    y_batch.cpu().numpy(), pred_train.detach().cpu().numpy()
                )
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], Step [{total_steps}], Average Loss: {avg_loss:.4f}, C-Index: {c_index:.4f}"
                )
                running_loss = 0.0  # Reset running loss after printing

                # Evaluate on validation set
                model.eval()
                eval_preds = []
                eval_labels = []
                with torch.no_grad():
                    for x_categ_batch, x_cont_batch, y_batch in val_loader:
                        pred_val = model(x_categ_batch, x_cont_batch)
                        eval_preds.append(pred_val.cpu().numpy())
                        eval_labels.append(y_batch.cpu().numpy())

                eval_preds = np.concatenate(eval_preds)
                eval_labels = np.concatenate(eval_labels)
                eval_loss = mean_squared_error(eval_labels, eval_preds)
                eval_metrics = compute_metrics((eval_preds, eval_labels))
                wandb.log(
                    {
                        "epoch": epoch + 1,
                        "eval_loss": eval_loss,
                        **{f"eval/{k}": v for k, v in eval_metrics.items()},
                    }
                )

                print(f"Validation Loss: {eval_loss:.4f}, Metrics: {eval_metrics}")

                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    early_stopping_counter = 0
                    top_models = save_model(model, eval_loss, top_models, run_name)
                else:
                    early_stopping_counter += 1
                    if early_stopping_counter >= early_stopping_steps:
                        print(f"Early stopping at epoch {epoch + 1}")
                        break

    ### test
    # load the best model and evaluate on the test set
    best_model_path = top_models[0][1]
    model.load_state_dict(torch.load(best_model_path))

    model.eval()
    test_preds = []
    test_labels = []

    with torch.no_grad():
        for x_categ_batch, x_cont_batch, y_batch in test_loader:
            pred_test = model(x_categ_batch, x_cont_batch)
            test_preds.append(pred_test.cpu().numpy())
            test_labels.append(y_batch.cpu().numpy())

    # calculate and print the final evaluation metrics
    test_preds = np.concatenate(test_preds)
    test_labels = np.concatenate(test_labels)
    test_loss = mean_squared_error(test_labels, test_preds)
    test_metrics = compute_metrics((test_preds, test_labels))

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Metrics: {test_metrics}")
    wandb.log(
        {"test_loss": test_loss, **{f"test/{k}": v for k, v in test_metrics.items()}}
    )
