"""
We improved the existing TabTransformer implementation to accommodate high-cardinality
categorical variables (like drugs) using embeddings from a language model.
We named it CatTransformer (Categorical Transformer Capable of High-Cardinality Variables).


References:
    - https://arxiv.org/pdf/2012.06678
    - https://github.com/lucidrains/tab-transformer-pytorch/blob/43ecec9740c4eaea2850b893692c77a0ad1e2695/tab_transformer_pytorch/tab_transformer_pytorch.py
"""

__author__ = "hw56@indiana.edu"
__version__ = "0.0.1"
__license__ = "0BSD"

import os
import pickle
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


class CatTransformer(nn.Module):
    def __init__(
            self,
            *,
            categories: list,  # a list containing the number of unique values (i.e., levels) within each easily encodable category (categories of low cardinality)
            num_high_card_categories: int,  # number of high-cardinality variables whose levels need to be derived from an LM; if set to 0, if falls back to a TabTransformer (https://arxiv.org/abs/2012.06678)
            num_continuous: int,  # number of continuous variables
            dim: int,  # input dimension/embedding size, paper set at 32
            depth: int = 6,  # number of stacking transformer blocks, paper recommended 6
            heads: int = 8,  # number of attention heads, paper recommends 8
            dim_head: int = 16,  # vector length for each attention head
            dim_out: int = 1,  # output dimension, fixed to 1 for regression
            mlp_hidden_mults: tuple = (4, 2),  # defines number of hidden layers of final MLP and multiplier of (bottom to top) input_size of (dim * num_categories) + num_continuous + dim
            mlp_act: nn.Module = nn.SiLU(),  # activation function for MLP
            continuous_mean_std: torch.Tensor = None,  # precomputed mean/std for continuous variables
            transformer_dropout: float = 0.1,  # dropout for attention and residual links
            use_shared_categ_embed: bool = True,  # share a fixed-length embeddings indicating the levels from the same column
            shared_categ_dim_divisor: int = 8,  # 1/8 of cat_embedding dims are shared in CatTransformer
            lm_model_name: str = 'distilbert/distilbert-base-uncased',  # Hugging Face BERT variant model name, and we recommend `Su-informatics-lab/gatortron_base_rxnorm_babbage_v2`
            lm_max_length: int = 512,  # max tokens for LM embedding computation
            embeddings_cache_path: str = '.lm_embeddings.pkl'  # path to cache embeddings
    ) -> None:
        super().__init__()
        assert all(map(lambda n: n > 0,
                       categories)), 'number of each category must be positive'
        assert len(
            categories) + num_continuous + num_high_card_categories > 0, 'input shape must not be null'

        self.num_categories = len(categories)
        self.num_high_card_categories = num_high_card_categories
        self.num_unique_categories = sum(categories)
        self.use_lm_embeddings = lm_model_name is not None and num_high_card_categories > 0
        self.dim = dim
        self.lm_max_length = lm_max_length

        total_tokens = self.num_unique_categories + 1
        shared_embed_dim = 0 if not use_shared_categ_embed else int(
            dim // shared_categ_dim_divisor)

        self.category_embed = nn.Embedding(total_tokens, dim - shared_embed_dim)
        self.use_shared_categ_embed = use_shared_categ_embed

        if use_shared_categ_embed:
            self.shared_category_embed = nn.Parameter(
                torch.zeros(self.num_categories, shared_embed_dim))
            nn.init.normal_(self.shared_category_embed, std=0.02)

        if self.num_unique_categories > 0:
            categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value=1)
            categories_offset = categories_offset.cumsum(dim=-1)[:-1]
            self.register_buffer('categories_offset', categories_offset)

        self.num_continuous = num_continuous

        if self.num_continuous > 0:
            if continuous_mean_std is not None:
                assert continuous_mean_std.shape == (num_continuous, 2), (
                    f'continuous_mean_std must have a shape of ({num_continuous}, 2) where the last '
                    f'dimension contains the mean and variance respectively')
            self.register_buffer('continuous_mean_std', continuous_mean_std)
            self.cont_norm = nn.LayerNorm(num_continuous)

        encoder_layers = TransformerEncoderLayer(d_model=dim, nhead=heads,
                                                 dim_feedforward=dim * dim_head,
                                                 dropout=transformer_dropout)
        self.transformer = TransformerEncoder(encoder_layers, num_layers=depth)

        transformer_input_size = dim * (self.num_categories + num_high_card_categories)
        input_size = transformer_input_size + num_continuous
        hidden_dimensions = [input_size * t for t in mlp_hidden_mults]
        all_dimensions = [input_size, *hidden_dimensions, dim_out]

        self.mlp = MLP(all_dimensions, act=mlp_act)

        if self.use_lm_embeddings:
            self.lm_model, self.tokenizer = self.load_lm_model(lm_model_name)
            self.embeddings_cache_path = embeddings_cache_path
            self.embeddings_cache = self.load_embeddings_cache()

    def forward(self, x_categ: torch.Tensor, x_cont: torch.Tensor,
                x_high_card_categ: list = None) -> torch.Tensor:
        device = x_categ.device
        self.categories_offset = self.categories_offset.to(device)

        assert x_categ.shape[
                   1] == self.num_categories, f'You must pass in {self.num_categories} values for your categories input'
        assert x_cont.shape[
                   1] == self.num_continuous, f'You must pass in {self.num_continuous} values for your continuous input'

        if self.num_high_card_categories > 0:
            assert x_high_card_categ is not None, 'You must pass in high-cardinality category values'
            assert len(x_high_card_categ) == x_categ.shape[
                0], 'The batch size of high-cardinality features must match the categorical features batch size'
        else:
            assert x_high_card_categ is None, 'High-cardinality category values should be None when num_high_card_categories is 0'

        x_categ = self.handle_missing_data(x_categ)

        if self.num_unique_categories > 0:
            x_categ = x_categ + self.categories_offset
            categ_embed = self.category_embed(x_categ)

            if self.use_shared_categ_embed:
                shared_categ_embed = self.shared_category_embed.unsqueeze(0).repeat(
                    categ_embed.shape[0], 1, 1)
                shared_categ_embed = shared_categ_embed.to(device)
                categ_embed = torch.cat((categ_embed, shared_categ_embed), dim=-1)

        if self.use_lm_embeddings and self.num_high_card_categories > 0:
            lm_cat_proj = self.get_lm_embeddings(x_high_card_categ, device)
            categ_embed = torch.cat((categ_embed, lm_cat_proj), dim=1)

        x = self.transformer(categ_embed)
        flat_categ = x.flatten(start_dim=1)

        if self.num_continuous > 0:
            if self.continuous_mean_std is not None:
                mean, std = self.continuous_mean_std.unbind(dim=-1)
                x_cont = (x_cont - mean) / std

            normed_cont = self.cont_norm(x_cont)
            x = torch.cat((flat_categ, normed_cont), dim=1)
        else:
            x = flat_categ

        logits = self.mlp(x)

        return logits

    def handle_missing_data(self, x_categ: torch.Tensor) -> torch.Tensor:
        missing_data_mask = (x_categ == 'na') | (x_categ == 'missing') | (
                x_categ == np.nan) | (x_categ == 'NaN') | (x_categ == 'N/A')
        x_categ[missing_data_mask] = 0
        return x_categ

    def load_lm_model(self, lm_model_name) -> tuple:
        model = AutoModel.from_pretrained(lm_model_name)
        tokenizer = AutoTokenizer.from_pretrained(lm_model_name)
        return model, tokenizer

    def load_embeddings_cache(self):
        if os.path.exists(self.embeddings_cache_path):
            with open(self.embeddings_cache_path, 'rb') as f:
                return pickle.load(f)
        return {}

    def save_embeddings_cache(self) -> None:
        with open(self.embeddings_cache_path, 'wb') as f:
            pickle.dump(self.embeddings_cache, f)

    def get_lm_embeddings(self, x_high_card_categ: list,
                          device: torch.device) -> torch.Tensor:
        new_texts = []

        for texts in x_high_card_categ:
            for text in texts:
                if text not in self.embeddings_cache:
                    new_texts.append(text)

        if new_texts:
            new_embeddings = self.compute_embeddings(new_texts, device)
            for text, embedding in zip(new_texts, new_embeddings):
                self.embeddings_cache[text] = embedding

            self.save_embeddings_cache()

        embeddings = [[self.embeddings_cache[text] for text in texts] for texts in
                      x_high_card_categ]
        embeddings = np.array(embeddings)
        embeddings = torch.tensor(embeddings, dtype=torch.float32).to(device)
        return embeddings

    def compute_embeddings(self, texts: list, device: torch.device) -> torch.Tensor:
        embeddings = []
        for text in tqdm(texts, desc="Computing LM embeddings"):
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True,
                                    padding=True, max_length=self.lm_max_length).to(
                device)
            outputs = self.lm_model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()
            embeddings.append(cls_embedding)

        embeddings = np.vstack(embeddings)
        pca = PCA(n_components=self.dim)
        reduced_embeddings = pca.fit_transform(embeddings)

        return torch.tensor(reduced_embeddings, dtype=torch.float32).to(device)

class MLP(nn.Module):
    def __init__(self, dims, act=nn.SiLU()) -> None:
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(act)
        self.mlp = nn.Sequential(*layers)

    def forward(self, x) -> torch.Tensor:
        return self.mlp(x)


class CatTransformerDataset(Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 categorical_features: List[str],
                 continuous_features: List[str],
                 pred_vars: List[str],
                 high_card_features: Optional[List[str]] = []) -> None:
        """
        Initializes the CatTransformerDataset.

        Args:
            df: The input DataFrame containing all the data.
            categorical_features: List of column names for categorical features.
            continuous_features: List of column names for continuous features.
            pred_vars: List of column names for prediction target variables.
            high_card_features: List of column names for high cardinality features.

        Returns:
            None
        """
        self.categorical_data = torch.tensor(df[categorical_features].values, dtype=torch.long)
        self.continuous_data = torch.tensor(df[continuous_features].values, dtype=torch.float)
        self.target_data = torch.tensor(df[pred_vars].values, dtype=torch.float)
        self.high_card_features = high_card_features

        if high_card_features:
            self.high_card_data = df[high_card_features].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.categorical_data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Optional[List[str]], torch.Tensor]:
        """
        Retrieves a single sample from the dataset.

        Args:
            idx: The index of the sample to retrieve.
        Returns:
            A tuple of:
                - Categorical data tensor
                - Continuous data tensor
                - List of high cardinality feature values (if present)
                - Target data tensor
        """
        if self.high_card_features:
            high_card_sample = self.high_card_data.iloc[idx].tolist()
            return (self.categorical_data[idx],
                    self.continuous_data[idx],
                    high_card_sample,
                    self.target_data[idx])
        else:
            return (self.categorical_data[idx],
                    self.continuous_data[idx],
                    None,
                    self.target_data[idx])
