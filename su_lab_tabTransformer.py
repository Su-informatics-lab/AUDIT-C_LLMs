"""
We can modify the existing TabTransformer implementation to accommodate high-cardinality
categorical variables (like drugs) using embeddings from a language model.

References:
    - https://arxiv.org/pdf/2012.06678
    - https://github.com/lucidrains/tab-transformer-pytorch/blob/43ecec9740c4eaea2850b893692c77a0ad1e2695/tab_transformer_pytorch/tab_transformer_pytorch.py
"""

import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from transformers import AutoModel, AutoTokenizer


class TabTransformer(nn.Module):
    def __init__(
            self,
            *,
            categories,  # tuple containing the number of unique values (i.e., levels) within each easily encodable category (categories of low cardinality)
            num_high_card_categories,  # number of high-cardinality variables whose levels need to be derived from an LM
            num_continuous,  # number of continuous variables
            dim,  # input dimension/embedding size, paper set at 32
            depth=6,  # number of stacking transformer blocks, paper recommended 6
            heads=8,  # number of attention heads, paper recommends 8
            dim_head=16,  # vector length for each attention head
            dim_out=1,  # output dimension, fixed to 1 for regression
            mlp_hidden_mults=(4, 2),  # defines number of hidden layers of final MLP and multiplier of (bottom to top) input_size of (dim * num_categories) + num_continuous + dim
            mlp_act=nn.SiLU(),  # activation function for MLP
            continuous_mean_std=None,
            attn_dropout=0.1,
            use_shared_categ_embed=True,
            shared_categ_dim_divisor=8,  # 1/8 of cat_embedding dims are shared
            lm_model_name=None,  # Hugging Face BERT variant model name
            embeddings_cache_path='.lm_embeddings.pkl'  # path to cache embeddings
    ):
        super().__init__()
        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'
        assert len(categories) + num_continuous + num_high_card_categories > 0, 'input shape must not be null'

        self.num_categories = len(categories)
        self.num_high_card_categories = num_high_card_categories
        self.num_unique_categories = sum(categories)
        self.use_lm_embeddings = lm_model_name is not None
        self.dim = dim

        total_tokens = self.num_unique_categories + 1  # for the missing value token
        shared_embed_dim = 0 if not use_shared_categ_embed else int(dim // shared_categ_dim_divisor)

        self.category_embed = nn.Embedding(total_tokens, dim - shared_embed_dim)
        self.use_shared_categ_embed = use_shared_categ_embed

        if use_shared_categ_embed:
            self.shared_category_embed = nn.Parameter(torch.zeros(self.num_categories, shared_embed_dim))
            nn.init.normal_(self.shared_category_embed, std=0.02)

        if self.num_unique_categories > 0:
            categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value=1)  # offset by 1 for the missing token
            categories_offset = categories_offset.cumsum(dim=-1)[:-1]
            self.register_buffer('categories_offset', categories_offset)

        self.num_continuous = num_continuous

        if self.num_continuous > 0:
            if continuous_mean_std is not None:
                assert continuous_mean_std.shape == (num_continuous, 2), (f'continuous_mean_std must '
                                                                          f'have a shape'
                                                                          f' of ({num_continuous}, '
                                                                          f'2) where the last '
                                                                          f'dimension contains the '
                                                                          f'mean and variance '
                                                                          f'respectively')
            self.register_buffer('continuous_mean_std', continuous_mean_std)
            self.cont_norm = nn.LayerNorm(num_continuous)

        encoder_layers = TransformerEncoderLayer(d_model=dim, nhead=heads,
                                                 dim_feedforward=dim * dim_head,
                                                 dropout=attn_dropout)
        self.transformer = TransformerEncoder(encoder_layers, num_layers=depth)

        input_size = (dim * self.num_categories) + num_continuous + (dim * num_high_card_categories)
        hidden_dimensions = [input_size * t for t in mlp_hidden_mults]
        all_dimensions = [input_size, *hidden_dimensions, dim_out]

        self.mlp = MLP(all_dimensions, act=mlp_act)

        if self.use_lm_embeddings:
            self.lm_model, self.tokenizer = self.load_lm_model(lm_model_name)
            self.embeddings_cache_path = embeddings_cache_path
            self.embeddings_cache = self.load_embeddings_cache()

    def forward(self, x_categ, x_cont, x_high_card_categ):
        assert x_categ.shape[1] == self.num_categories, (f'you must pass in '
                                                         f'{self.num_categories} values for '
                                                         f'your categories input')
        assert x_cont.shape[1] == self.num_continuous, (f'you must pass in '
                                                        f'{self.num_continuous} values for '
                                                        f'your continuous input')
        assert x_high_card_categ.shape[1] == self.num_high_card_categories, (f'you must pass in '
                                                                            f'{self.num_high_card_categories} high-cardinality category values')

        # handle missing data in categorical variables
        x_categ = self.handle_missing_data(x_categ)

        if self.num_unique_categories > 0:
            x_categ = x_categ + self.categories_offset
            categ_embed = self.category_embed(x_categ)

            if self.use_shared_categ_embed:
                shared_categ_embed = self.shared_category_embed.unsqueeze(0).repeat(categ_embed.shape[0], 1, 1)
                categ_embed = torch.cat((categ_embed, shared_categ_embed), dim=-1)

            x = self.transformer(categ_embed)
            flat_categ = x.flatten(start_dim=1)

        if self.num_continuous > 0:
            if self.continuous_mean_std is not None:
                mean, std = self.continuous_mean_std.unbind(dim=-1)
                x_cont = (x_cont - mean) / std

            normed_cont = self.cont_norm(x_cont)

        # project LM embeddings to `dim` for high-cardinality categorical variables
        if self.use_lm_embeddings:
            lm_cat_proj = self.get_lm_embeddings(x_high_card_categ)

        x = torch.cat((flat_categ, normed_cont, lm_cat_proj), dim=1)
        logits = self.mlp(x)

        return logits

    def handle_missing_data(self, x_categ):
        # identify missing data and replace with the index of the special token
        missing_data_mask = (x_categ == 'na') | (x_categ == 'missing') | (x_categ == np.nan) | (x_categ == 'NaN') | (x_categ == 'N/A')
        x_categ[missing_data_mask] = 0  # 0 is the index of the special token for missing data
        return x_categ

    def load_lm_model(self, lm_model_name):
        model = AutoModel.from_pretrained(lm_model_name)
        tokenizer = AutoTokenizer.from_pretrained(lm_model_name)
        return model, tokenizer

    def load_embeddings_cache(self):
        if os.path.exists(self.embeddings_cache_path):
            with open(self.embeddings_cache_path, 'rb') as f:
                return pickle.load(f)
        return {}

    def save_embeddings_cache(self):
        with open(self.embeddings_cache_path, 'wb') as f:
            pickle.dump(self.embeddings_cache, f)

    def get_lm_embeddings(self, x_high_card_categ):
        embeddings = []
        new_texts = []

        for texts in x_high_card_categ.values:
            embeddings_for_row = []
            for text in texts:
                if text in self.embeddings_cache:
                    embeddings_for_row.append(self.embeddings_cache[text])
                else:
                    new_texts.append(text)
            embeddings.append(embeddings_for_row)

        if new_texts:
            new_embeddings = self.compute_embeddings(new_texts)
            for text, embedding in zip(new_texts, new_embeddings):
                self.embeddings_cache[text] = embedding

            # save updated cache
            self.save_embeddings_cache()

        # convert to tensor
        embeddings = [[self.embeddings_cache[text] for text in texts] for texts in x_high_card_categ.values]
        return torch.tensor(embeddings, dtype=torch.float32)

    def compute_embeddings(self, texts):
        inputs = self.tokenizer(texts, return_tensors='pt', truncation=True, padding=True, max_length=512)
        outputs = self.lm_model(**inputs)
        cls_embeddings = outputs.last_hidden_state[:, 0, :].detach().numpy()  # fixme: only support [CLS] token embedding for now

        # perform PCA to reduce dimensionality
        pca = PCA(n_components=self.dim)
        reduced_embeddings = pca.fit_transform(cls_embeddings)

        return torch.tensor(reduced_embeddings, dtype=torch.float32)


class MLP(nn.Module):
    def __init__(self, dims, act=nn.SiLU()):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2):
                layers.append(act)
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)
