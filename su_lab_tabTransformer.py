"""
Using a language model's embedding for high-cardinality categorical variables
(like drugs), we can modify the existing TabTransformer implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class TabTransformer(nn.Module):
    def __init__(
            self,
            *,
            categories,
            num_continuous,
            dim,
            depth,
            heads,
            dim_head=16,
            dim_out=1,
            mlp_hidden_mults=(4, 2),
            mlp_act=nn.ReLU(),
            num_special_tokens=0,
            continuous_mean_std=None,
            attn_dropout=0.1,
            use_shared_categ_embed=True,
            shared_categ_dim_divisor=8  # 1/8 of cat_embedding dims are shared
    ):
        super().__init__()
        assert all(map(lambda n: n > 0,
                       categories)), 'number of each category must be positive'
        assert len(categories) + num_continuous > 0, 'input shape must not be null'

        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)

        total_tokens = self.num_unique_categories + num_special_tokens
        shared_embed_dim = 0 if not use_shared_categ_embed else int(
            dim // shared_categ_dim_divisor)

        self.category_embed = nn.Embedding(total_tokens, dim - shared_embed_dim)
        self.use_shared_categ_embed = use_shared_categ_embed

        if use_shared_categ_embed:
            self.shared_category_embed = nn.Parameter(
                torch.zeros(self.num_categories, shared_embed_dim))
            nn.init.normal_(self.shared_category_embed, std=0.02)

        if self.num_unique_categories > 0:
            categories_offset = F.pad(torch.tensor(list(categories)), (1, 0),
                                      value=num_special_tokens)
            categories_offset = categories_offset.cumsum(dim=-1)[:-1]
            self.register_buffer('categories_offset', categories_offset)

        self.num_continuous = num_continuous

        if self.num_continuous > 0:
            if continuous_mean_std is not None:
                assert continuous_mean_std.shape == (num_continuous,
                                                     2), (f'continuous_mean_std must '
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

        input_size = (dim * self.num_categories) + num_continuous
        hidden_dimensions = [input_size * t for t in mlp_hidden_mults]
        all_dimensions = [input_size, *hidden_dimensions, dim_out]

        self.mlp = MLP(all_dimensions, act=mlp_act)

    def forward(self, x_categ, x_cont):
        assert x_categ.shape[
                   1] == self.num_categories, (f'you must pass in '
                                               f'{self.num_categories} values for '
                                               f'your categories input')
        assert x_cont.shape[
                   1] == self.num_continuous, (f'you must pass in '
                                               f'{self.num_continuous} values for '
                                               f'your continuous input')

        if self.num_unique_categories > 0:
            x_categ = x_categ + self.categories_offset
            categ_embed = self.category_embed(x_categ)

            if self.use_shared_categ_embed:
                shared_categ_embed = self.shared_category_embed.unsqueeze(0).repeat(
                    categ_embed.shape[0], 1, 1)
                categ_embed = torch.cat((categ_embed, shared_categ_embed), dim=-1)

            x = self.transformer(categ_embed)
            flat_categ = x.flatten(start_dim=1)

        if self.num_continuous > 0:
            if self.continuous_mean_std is not None:
                mean, std = self.continuous_mean_std.unbind(dim=-1)
                x_cont = (x_cont - mean) / std

            normed_cont = self.cont_norm(x_cont)

        x = torch.cat((flat_categ, normed_cont), dim=1)
        logits = self.mlp(x)

        return logits


class MLP(nn.Module):
    def __init__(self, dims, act=nn.SiLU()):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(act)
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)
