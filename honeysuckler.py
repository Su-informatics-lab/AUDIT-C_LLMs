import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoConfig
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import random
from utils import SEED
import os

def generate_drug_embeddings(
        df: pd.DataFrame,
        text_column: str,
        person_id_column: str,
        model_name: str = 'UFNLP/gatortron-base',
        embedding_dim: int = 32,
        batch_size: int = 32,
        random_state: int = 42,
        reduction_method: str = 'pca',  # options: 'pca' or 'autoencoder'
        device: str = None,
        cache_file: str = None
    ) -> pd.DataFrame:
    """
    Generate embeddings from drug use text data using a specified language model and
    reduce dimensionality using PCA or an autoencoder.

    Parameters:
        df: pandas DataFrame containing the data.
        text_column: Name of the column containing the drug use text.
        person_id_column: Name of the column containing the unique identifier (e.g., 'person_id').
        model_name: Name of the Hugging Face model to use.
        embedding_dim: Desired dimension for the output embeddings.
        batch_size: Batch size for processing data.
        random_state: Seed for reproducibility.
        reduction_method: Method for dimensionality reduction ('pca' or 'autoencoder').
        device: Device to run the model on ('cpu' or 'cuda'). If None, automatically
            selects GPU if available.
        cache_file: Path to the cache file storing raw embeddings.

    Returns:
        A DataFrame containing the reduced embeddings, indexed by person_id.
    """
    # sanity checks
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in the DataFrame.")
    if person_id_column not in df.columns:
        raise ValueError(f"Column '{person_id_column}' not found in the DataFrame.")

    # set device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # prepare the cache
    cache_exists = cache_file is not None and os.path.exists(cache_file)
    if cache_exists:
        cached_embeddings = pd.read_parquet(cache_file)
        cached_person_ids = set(cached_embeddings.index)
    else:
        cached_embeddings = pd.DataFrame()
        cached_person_ids = set()

    # identify person_ids needing embeddings
    all_person_ids = set(df[person_id_column])
    person_ids_to_compute = all_person_ids - cached_person_ids

    # prepare data for embedding generation
    df_to_embed = df[df[person_id_column].isin(person_ids_to_compute)].copy()

    # replace missing values with a placeholder and ensure all entries are strings
    df_to_embed[text_column] = df_to_embed[text_column].fillna('no_drug').astype(str)
    texts = df_to_embed[text_column].tolist()
    person_ids = df_to_embed[person_id_column].tolist()

    # load tokenizer and model if needed
    if person_ids_to_compute:
        # load tokenizer and model
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            config = AutoConfig.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name, config=config)
        except Exception as e:
            raise ValueError(f"Error loading model '{model_name}': {e}")

        model.to(device)
        model.eval()

        # identify model type
        model_type = config.model_type
        is_encoder_decoder = config.is_encoder_decoder

        # generate embeddings
        embeddings = []
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Generating Embeddings"):
                batch_texts = texts[i:i + batch_size]
                encoding = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                )
                input_ids = encoding['input_ids'].to(device)
                attention_mask = encoding['attention_mask'].to(device)

                # get model outputs
                if is_encoder_decoder:
                    # for encoder-decoder models, use encoder outputs
                    encoder_outputs = model.encoder(input_ids=input_ids,
                                                    attention_mask=attention_mask)
                    last_hidden_state = encoder_outputs.last_hidden_state  # (batch_size, seq_length, hidden_size)
                    # pooling: take the mean of the encoder outputs
                    pooled_output = last_hidden_state.mean(dim=1)  # (batch_size, hidden_size)
                elif model_type in ['megatron-bert', 'bert', 'roberta', 'distilbert', 'albert']:
                    # for encoder models, use the [CLS] token embedding
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    last_hidden_state = outputs.last_hidden_state  # (batch_size, seq_length, hidden_size)
                    pooled_output = last_hidden_state[:, 0, :]  # (batch_size, hidden_size)
                elif model_type in ['gpt2', 'gpt', 'xlm']:
                    # for decoder-only models, use the last hidden state
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    last_hidden_state = outputs.last_hidden_state  # (batch_size, seq_length, hidden_size)
                    # pooling: take the mean of the last hidden states
                    pooled_output = last_hidden_state.mean(dim=1)  # (batch_size, hidden_size)
                else:
                    raise ValueError(f"Model type '{model_type}' is not supported.")

                embeddings.append(pooled_output.cpu().numpy())

        # stack all new embeddings
        embeddings = np.concatenate(embeddings, axis=0)  # (num_new_samples, hidden_size)

        # create DataFrame for new embeddings
        embeddings_new_df = pd.DataFrame(
            embeddings,
            index=df_to_embed[person_id_column],
            columns=[f'hidden_{i}' for i in range(embeddings.shape[1])]
        )

        # update the cache
        if cache_file is not None:
            # combine with cached embeddings
            embeddings_raw_df = pd.concat([cached_embeddings, embeddings_new_df], axis=0)
            # save updated cache
            embeddings_raw_df.to_parquet(cache_file, index=True)
            print(f"Cache updated at {cache_file}")
        else:
            embeddings_raw_df = embeddings_new_df
    else:
        # no new embeddings needed; use cache only
        embeddings_raw_df = cached_embeddings

    # ensure embeddings are aligned with df
    embeddings_raw_df = embeddings_raw_df.loc[df[person_id_column]]

    # proceed with dimensionality reduction
    embeddings_array = embeddings_raw_df.values  # (num_samples, hidden_size)

    # dimensionality reduction: pca or autoencoder
    if reduction_method == 'pca':
        pca = PCA(n_components=embedding_dim, random_state=random_state)
        embeddings_reduced = pca.fit_transform(embeddings_array)  # (num_samples, embedding_dim)
    elif reduction_method == 'autoencoder':
        # Define the autoencoder model
        class Autoencoder(nn.Module):
            def __init__(self, input_dim, latent_dim):
                super(Autoencoder, self).__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, latent_dim * 2),
                    nn.SiLU(),
                    nn.Linear(latent_dim * 2, latent_dim)
                )
                self.decoder = nn.Sequential(
                    nn.Linear(latent_dim, latent_dim * 2),
                    nn.SiLU(),
                    nn.Linear(latent_dim * 2, input_dim)
                )

            def forward(self, x):
                z = self.encoder(x)
                reconstructed = self.decoder(z)
                return reconstructed, z

        input_dim = embeddings_array.shape[1]
        latent_dim = embedding_dim
        autoencoder = Autoencoder(input_dim, latent_dim).to(device)

        # training parameters
        num_epochs = 500
        learning_rate = 3e-4
        patience = 10  # for early stopping

        # split data into training and validation sets
        X_train, X_val = train_test_split(
            embeddings_array,
            test_size=0.2,
            random_state=random_state
        )

        # prepare data loaders
        train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)

        train_dataset = torch.utils.data.TensorDataset(train_tensor)
        val_dataset = torch.utils.data.TensorDataset(val_tensor)

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=False)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)

        # early stopping variables
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_model_state = None

        # training loop with early stopping
        for epoch in range(num_epochs):
            autoencoder.train()
            total_loss = 0
            for data_batch in train_loader:
                inputs = data_batch[0]
                optimizer.zero_grad()
                reconstructed, _ = autoencoder(inputs)
                loss = criterion(reconstructed, inputs)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * inputs.size(0)
            avg_train_loss = total_loss / len(train_loader.dataset)

            # validation
            autoencoder.eval()
            val_loss = 0
            with torch.no_grad():
                for data_batch in val_loader:
                    inputs = data_batch[0]
                    reconstructed, _ = autoencoder(inputs)
                    loss = criterion(reconstructed, inputs)
                    val_loss += loss.item() * inputs.size(0)
            avg_val_loss = val_loss / len(val_loader.dataset)

            # check for improvement
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                best_model_state = autoencoder.state_dict()
            else:
                epochs_no_improve += 1

            # early stopping
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.8f}, Val Loss: {avg_val_loss:.8f}")

        # load the best model
        autoencoder.load_state_dict(best_model_state)

        # obtain the reduced embeddings
        autoencoder.eval()
        with torch.no_grad():
            embeddings_tensor = torch.tensor(embeddings_array, dtype=torch.float32).to(device)
            _, embeddings_reduced = autoencoder(embeddings_tensor)
            embeddings_reduced = embeddings_reduced.cpu().numpy()
    else:
        raise ValueError(
            f"Reduction method '{reduction_method}' is not supported. "
            f"Choose 'pca' or 'autoencoder'.")

    # create a df for the embeddings
    embedding_columns = [f'embed_{i}' for i in range(embedding_dim)]
    embeddings_df = pd.DataFrame(
        embeddings_reduced,
        columns=embedding_columns,
        index=df[person_id_column]
    )

    # ensure embeddings_df index is aligned with df index
    embeddings_df = embeddings_df.set_index(df.index)

    return embeddings_df

if __name__ == "__main__":
    # set random seeds for reproducibility
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    parser = argparse.ArgumentParser(
        description="Produce drug vectors for downstream tasks"
    )
    parser.add_argument(
        "--input_file", required=True, type=str,
        help="Path to the input parquet containing the drug use text data."
    )
    parser.add_argument(
        "--text_column", default='standard_concept_name', type=str,
        help="Name of the column containing the drug use text."
    )
    parser.add_argument(
        "--person_id_column", default='person_id', type=str,
        help="Name of the column containing the unique identifier."
    )
    parser.add_argument(
        "--model_name", default='UFNLP/gatortron-base', type=str,
        help="Name of the Hugging Face model to use."
    )
    parser.add_argument(
        "--embedding_dim", default=32, type=int,
        help="Desired dimension for the output embeddings."
    )
    parser.add_argument(
        "--batch_size", default=32, type=int,
        help="Batch size for processing data."
    )
    parser.add_argument(
        "--reduction_method",
        default='pca', choices=['pca', 'autoencoder'], type=str,
        help="Method for dimensionality reduction ('pca' or 'autoencoder')."
    )
    parser.add_argument(
        "--output_file", required=True, type=str,
        help="Path to save the output embeddings file (Parquet format)."
    )
    parser.add_argument(
        "--device", default=None, type=str,
        help="Device to run the model on ('cpu' or 'cuda'). If None, automatically "
             "selects GPU if available."
    )
    parser.add_argument(
        "--cache_file",
        default='embeddings/.gatortron_base_embed_cache.parquet', type=str,
        help="Path to the cache file storing raw embeddings."
    )
    args = parser.parse_args()

    # load the input data
    df = pd.read_parquet(args.input_file)

    # generate embeddings
    embeddings_df = generate_drug_embeddings(
        df=df,
        text_column=args.text_column,
        person_id_column=args.person_id_column,
        model_name=args.model_name,
        embedding_dim=args.embedding_dim,
        batch_size=args.batch_size,
        reduction_method=args.reduction_method,
        device=args.device,
        cache_file=args.cache_file,
        random_state=SEED
    )

    # save the embeddings to a Parquet file, keeping the original index
    embeddings_df.to_parquet(args.output_file, index=True)
    print(f"Embeddings saved to {args.output_file}")
