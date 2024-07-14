import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
from lifelines.utils import concordance_index
from sklearn.metrics import mean_squared_error
from torch import optim
from torch.utils.data import DataLoader, RandomSampler, TensorDataset

from tabtransformer import TabTransformer
from utils import PROJECT_NAME, SEED, compute_metrics

# Initialize wandb
wandb.init(project=PROJECT_NAME, name="your_run_name")

# for reproducibility
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def save_model(model, eval_loss, top_models):
    model_path = f"model_{eval_loss:.4f}.pt"
    torch.save(model.state_dict(), model_path)
    top_models.append((eval_loss, model_path))
    top_models = sorted(top_models, key=lambda x: x[0])[:3]  # Keep only top 3 models
    return top_models


if __name__ == "__main__":
    ### prepare data
    AUDIT_C_SCORING_212K_ALL_FEATURES_W_ACCUMULATED_LOG2FC_PATH = (
        "gs://fc-secure-19ab668e-266f-4a5f-9c63-febea17b23cf/data/hw56"
        "/AUD_LLM_DEMO_COMO_212K_ALL_FEATURES_W_ACCUMULATED_LOG2FC.parquet"
    )
    df = pd.read_parquet(AUDIT_C_SCORING_212K_ALL_FEATURES_W_ACCUMULATED_LOG2FC_PATH)

    # reserve data we used for this run
    demoComo = [
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
        "age",
        "audit.c.score",
    ]
    assert len(demoComo) == 22 + 1
    # df_demoComo = df[demoComo]

    df_demoComo_train = df.loc[df["split"] == "train"][demoComo]
    df_demoComo_test = df.loc[df["split"] == "test"][demoComo]

    # Convert the train and test sets to tensors
    x_categ_train = torch.tensor(
        df_demoComo_train[demoComo[:-2]].values, dtype=torch.long
    ).to(device)
    x_cont_train = torch.tensor(
        df_demoComo_train[["age"]].values, dtype=torch.float
    ).to(device)
    y_train = (
        torch.tensor(df_demoComo_train["audit.c.score"].values, dtype=torch.float)
        .unsqueeze(1)
        .to(device)
    )

    x_categ_test = torch.tensor(
        df_demoComo_test[demoComo[:-2]].values, dtype=torch.long
    ).to(device)
    x_cont_test = torch.tensor(df_demoComo_test[["age"]].values, dtype=torch.float).to(
        device
    )
    y_test = (
        torch.tensor(df_demoComo_test["audit.c.score"].values, dtype=torch.float)
        .unsqueeze(1)
        .to(device)
    )

    cont_mean_std = (
        torch.Tensor([df_demoComo_train["age"].mean(), df_demoComo_train["age"].std()])
        .reshape(-1, 2)
        .to(device)
    )

    # find out unique values in each column for categorical values
    categories = []
    for c in df_demoComo_train.columns[:-2]:
        categories.append(len(df_demoComo_train[c].unique()))
    assert len(categories) == 21  # 21 cats
    ###

    model = TabTransformer(
        categories=categories,  # tuple containing the number of unique values within each category
        num_continuous=1,  # number of continuous values
        dim=32,  # input/cat_embedding dimension, paper set at 32
        depth=6,  # num transformer layers, paper recommended 6
        heads=8,  # num heads, paper recommends 8
        dim_head=16,  # num heads
        dim_out=1,  # out dim, but could be anything
        mlp_hidden_mults=(4,2),  # relative multiples of each hidden dimension of the last mlp to logits
        mlp_act=nn.SiLU(),  # activation for final mlp, defaults to relu, but could be anything else (selu etc)
        num_special_tokens=0,
        continuous_mean_std=cont_mean_std,
        attn_dropout=0.1,
        ff_dropout=0.1,
        use_shared_categ_embed=True,
        shared_categ_dim_divisor=8,  # 1/8 of cat_embedding dims are shared
    ).to(device)

    # training settings
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    num_epochs = 100
    batch_size = 16

    train_dataset = TensorDataset(x_categ_train, x_cont_train, y_train)
    test_dataset = TensorDataset(x_categ_test, x_cont_test, y_test)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=RandomSampler(train_dataset)
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # early stopping and model saving settings
    early_stopping_steps = 50
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

            if total_steps % 1000 == 0:
                avg_loss = running_loss / 1000
                c_index = concordance_index(
                    y_batch.cpu().numpy(), pred_train.detach().cpu().numpy()
                )
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], Step [{total_steps}], Average Loss: {avg_loss:.4f}, C-Index: {c_index:.4f}"
                )
                running_loss = 0.0  # Reset running loss after printing

        model.eval()
        eval_preds = []
        eval_labels = []
        with torch.no_grad():
            for x_categ_batch, x_cont_batch, y_batch in test_loader:
                pred_test = model(x_categ_batch, x_cont_batch)
                eval_preds.append(pred_test.cpu().numpy())
                eval_labels.append(y_batch.cpu().numpy())

        eval_preds = np.concatenate(eval_preds)
        eval_labels = np.concatenate(eval_labels)
        eval_loss = mean_squared_error(eval_labels, eval_preds)
        eval_metrics = compute_metrics((eval_preds, eval_labels))
        wandb.log({"epoch": epoch + 1, "eval_loss": eval_loss, **eval_metrics})

        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            early_stopping_counter = 0
            top_models = save_model(model, eval_loss, top_models)
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
    wandb.log({"test_loss": test_loss, **test_metrics})

    # model = TabTransformer(
    #     categories=categories,
    #
    #     num_continuous=1,  # number of continuous values
    #     dim=32,  # dimension, paper set at 32
    #     dim_out=1,  # binary prediction, but could be anything
    #     depth=6,  # depth, paper recommended 6
    #     heads=8,  # heads, paper recommends 8
    #     attn_dropout=0.1,  # post-attention dropout
    #     ff_dropout=0.1,  # feed forward dropout
    #     mlp_hidden_mults=(4, 2),
    #     # relative multiples of each hidden dimension of the last mlp to logits
    #     mlp_act=nn.ReLU(),
    #     # activation for final mlp, defaults to relu, but could be anything else (selu etc)
    #     continuous_mean_std=cont_mean_std
    #     # (optional) - normalize the continuous values before layer norm
    # ).to(device)
