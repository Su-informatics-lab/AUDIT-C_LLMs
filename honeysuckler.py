import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoConfig
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.optim as optim
import random
from utils import SEED


def generate_drug_embeddings(
        df: pd.DataFrame,
        text_column: str,
        model_name: str = 'UFNLP/gatortron-base',
        embedding_dim: int = 32,
        batch_size: int = 32,
        random_state: int = 42,
        reduction_method: str = 'pca',  # options: 'pca' or 'autoencoder'
        device: str = None
) -> pd.DataFrame:
    """
    Generate embeddings from drug use text data using a specified language model and
    reduce dimensionality using PCA or an autoencoder.

    Parameters:
        df: pandas DataFrame containing the data.
        text_column: Name of the column containing the drug use text.
        model_name: Name of the Hugging Face model to use.
        embedding_dim: Desired dimension for the output embeddings.
        batch_size: Batch size for processing data.
        random_state: Seed for reproducibility.
        reduction_method: Method for dimensionality reduction ('pca' or 'autoencoder').
        device: Device to run the model on ('cpu' or 'cuda'). If None, automatically
        selects GPU if available.

    Returns:
        A df containing the reduced embeddings.
    """
    # sanity check
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in the DataFrame.")

    # set device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

    embeddings = []
    # replace missing values with an empty string and ensure all entries are strings
    df[text_column] = df[text_column].fillna('None').astype(str)
    texts = df[text_column].tolist()

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
                pooled_output = last_hidden_state.mean(
                    dim=1)  # (batch_size, hidden_size)
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
                pooled_output = last_hidden_state.mean(
                    dim=1)  # (batch_size, hidden_size)
            else:
                raise ValueError(f"Model type '{model_type}' is not supported.")

            embeddings.append(pooled_output.cpu().numpy())

    # stack all embeddings
    embeddings = np.concatenate(embeddings, axis=0)  # (num_samples, hidden_size)

    # dimensionality reduction: pca or autoencoder
    if reduction_method == 'pca':
        pca = PCA(n_components=embedding_dim, random_state=random_state)
        embeddings_reduced = pca.fit_transform(
            embeddings)  # (num_samples, embedding_dim)
    elif reduction_method == 'autoencoder':
        # define the autoencoder model
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

        input_dim = embeddings.shape[1]
        latent_dim = embedding_dim
        autoencoder = Autoencoder(input_dim, latent_dim).to(device)

        # training parameters
        num_epochs = 1000
        learning_rate = 3e-4

        # prepare data
        embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32).to(device)
        dataset = torch.utils.data.TensorDataset(embeddings_tensor)
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=batch_size,
                                                 shuffle=True)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)

        # training loop
        for epoch in range(num_epochs):
            total_loss = 0
            for data_batch in dataloader:
                inputs = data_batch[0]
                optimizer.zero_grad()
                reconstructed, _ = autoencoder(inputs)
                loss = criterion(reconstructed, inputs)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * inputs.size(0)
            avg_loss = total_loss / len(dataloader.dataset)
            # print loss per epoch
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

        # obtain the reduced embeddings
        with torch.no_grad():
            embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32).to(device)
            _, embeddings_reduced = autoencoder(embeddings_tensor)
            embeddings_reduced = embeddings_reduced.cpu().numpy()
    else:
        raise ValueError(
            f"Reduction method '{reduction_method}' is not supported. "
            f"Choose 'pca' or 'autoencoder'.")

    # create a DataFrame for the embeddings
    embedding_columns = [f'embed_{i}' for i in range(embedding_dim)]
    embeddings_df = pd.DataFrame(embeddings_reduced, columns=embedding_columns)

    # reset index to align with the original DataFrame
    embeddings_df.index = df.index

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
    args = parser.parse_args()

    # load the input data
    df = pd.read_parquet(args.input_file)

    # generate embeddings
    embeddings_df = generate_drug_embeddings(
        df=df,
        text_column=args.text_column,
        model_name=args.model_name,
        embedding_dim=args.embedding_dim,
        batch_size=args.batch_size,
        reduction_method=args.reduction_method,
        device=args.device,
        random_state=SEED
    )

    # save the embeddings to a Parquet file, keeping the original index
    embeddings_df.to_parquet(args.output_file, index=True)
    print(f"Embeddings saved to {args.output_file}")
