import torch
import torch.nn as nn

from smiles23d.modules.contval_encoders import TimeEncodingHead, encoder_factory
from smiles23d.modules.graph_encoders import graph_encoder_factory
from smiles23d.modules.masking import create_masks
from smiles23d.modules.structural_layers import StructureTransformerEncoder

from .moduleparams import ScoreNetParams


class Transformer(nn.Module):
    def __init__(
        self,
        layers: list[int],
        n_embd: int,
        n_head: int,
        vocab_size: int,
        n_aux_tokens: int,
        point_encoder_kwargs: dict,
        graph_encoder_kwargs: dict,
        n_time_basis: int = 10,
        translation_augmentation: float | None = None,
    ) -> None:
        super().__init__()

        pars = ScoreNetParams(
            layers,
            n_embd,
            n_head,
            vocab_size,
            n_aux_tokens,
            n_time_basis,
            translation_augmentation,
        )

        self.time_encoding_head = TimeEncodingHead(pars.n_time_basis)
        self.time_projection = nn.Linear(pars.n_time_basis, pars.n_embd, bias=False)
        self.score_head = ScoreHead(pars.n_embd)
        self.graph_encoder = graph_encoder_factory(graph_encoder_kwargs, pars)
        point_encoder = encoder_factory(point_encoder_kwargs)

        self.mixed_structural_decoder = None

        if layers[2]:
            self.structural_decoder = StructureTransformerEncoder(pars, point_encoder)
        else:
            self.structural_decoder = None
        self.register_parameter(
            "aux_tokens", param=nn.Parameter(torch.randn(n_aux_tokens, n_embd))
        )
        self.translation_augmentation = translation_augmentation
        self.n_aux_tokens = n_aux_tokens

    def create_masks(self, atoms, alibi):
        attention_mask_aux_tokens, attention_mask, padding_maks = create_masks(
            atoms, self.n_aux_tokens, alibi
        )
        return attention_mask_aux_tokens, attention_mask, padding_maks

    def encode_graph(self, features, attention_mask):
        h = self.graph_encoder(features, attention_mask)
        return h

    def decode_structure(self, h, features, attention_mask, t, dr):
        time_tokens = (
            self.aux_tokens[None, ...]
            + self.time_projection(self.time_encoding_head(t))[:, None, :]
        )
        h = torch.cat([time_tokens, h], dim=1)

        if self.structural_decoder is not None:
            h = self.structural_decoder(h, features, dr, attention_mask)
        score = h[:, self.n_aux_tokens :]
        score = self.score_head(score)
        return score

    def forward(
        self, features, attention_mask, attention_mask_aux_tokens, t, dr
    ) -> torch.Tensor:
        h = self.encode_graph(features, attention_mask)
        if self.translation_augmentation is not None and self.training:
            with torch.no_grad():
                B = dr.shape[0]
                dr = (
                    dr
                    + torch.rand(B, 1, 3, dtype=dr.dtype, device=dr.device)
                    * self.translation_augmentation
                )
        score = self.decode_structure(h, features, attention_mask_aux_tokens, t, dr)
        return score


class ScoreHead(nn.Module):
    def __init__(self, n_embd: int):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 3, bias=False)

    def forward(self, x) -> torch.Tensor:
        return self.c_fc(x)
