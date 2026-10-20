from functools import partial

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .moduleparams import ScoreNetParams


def _find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


def _calc_hidden_mlp_dim(n_input):
    hidden_dim = 4 * n_input
    n_hidden = int(2 * hidden_dim / 3)
    n_hidden = _find_multiple(n_hidden, 256)
    return n_hidden


class MLP(nn.Module):
    def __init__(self, n_embd: int) -> None:
        super().__init__()
        n_hidden = _calc_hidden_mlp_dim(n_embd)
        self.c_fc1 = nn.Linear(n_embd, n_hidden, bias=False)
        self.c_fc2 = nn.Linear(n_embd, n_hidden, bias=False)
        self.c_proj = nn.Linear(n_hidden, n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.c_fc1(x)) * self.c_fc2(x)
        x = self.c_proj(x)
        return x


class RMSNorm(nn.Module):
    def __init__(self, size: int, dim: int = -1, eps: float = 1e-5) -> None:
        super().__init__()
        self.register_parameter("scale", param=nn.Parameter(torch.ones(size)))
        self.eps = eps
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = x.pow(2).mean(dim=self.dim, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return (self.scale * x_normed).type_as(x)


class StructureAttentionBlock(nn.Module):
    def __init__(self, score_net_params: ScoreNetParams, point_encoder) -> None:
        super().__init__()

        self.score_net_params = score_net_params
        n_embd = score_net_params.n_embd
        n_head = score_net_params.n_head

        assert n_embd % n_head == 0
        self.spatial_proj = point_encoder()
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        features: dict[str, torch.Tensor],
        r: torch.Tensor,
        dr: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        (B, T, C) = x.size()

        n_embd = self.score_net_params.n_embd
        n_head = self.score_net_params.n_head
        n_aux_tokens = self.score_net_params.n_aux_tokens

        if self.spatial_proj is not None:
            if n_aux_tokens:
                x_without_aux = x[:, n_aux_tokens:, :]
                x_without_aux = x_without_aux + self.spatial_proj(
                    dr.view(B, T - n_aux_tokens, -1)
                )
                x = torch.cat([x[:, :n_aux_tokens, :], x_without_aux], dim=1)
            else:
                x = x + self.spatial_proj(dr.view(B, T - n_aux_tokens, -1))

        q, k, v = self.c_attn(x).split(n_embd, dim=2)

        head_size = C // n_head
        k = k.view(B, T, n_head, head_size).transpose(1, 2)
        q = q.view(B, T, n_head, head_size).transpose(1, 2)
        v = v.view(B, T, n_head, head_size).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)

        return y


class StructureEncoderTransformerBlock(nn.Module):
    def __init__(self, score_net_params: ScoreNetParams, point_encoder) -> None:
        super().__init__()

        n_embd = score_net_params.n_embd
        self.rms_1 = RMSNorm(n_embd)
        self.attn = StructureAttentionBlock(score_net_params, point_encoder)
        self.rms_2 = RMSNorm(n_embd)
        self.mlp = MLP(n_embd)

    def forward(
        self,
        x: torch.FloatTensor,
        features: dict[str, torch.Tensor],
        r: torch.FloatTensor,
        dr: torch.FloatTensor,
        attention_mask: torch.FloatTensor,
    ) -> torch.Tensor:
        x = x + self.attn(self.rms_1(x), features, r, dr, attention_mask)
        x = x + self.mlp(self.rms_2(x))
        return x


class StructureTransformerEncoder(nn.Module):
    def __init__(self, score_net_params: ScoreNetParams, point_encoder) -> None:
        super().__init__()

        n_layers = score_net_params.layers[2]

        self.point_encoder = point_encoder
        transformer_blocks = []
        for layer_index in range(n_layers):
            point_encoder = partial(
                self.point_encoder.build_encoder, layer_index, score_net_params.n_embd
            )
            transformer_blocks.append(
                StructureEncoderTransformerBlock(score_net_params, point_encoder)
            )
        self.blocks = nn.ModuleList(transformer_blocks)

        n_head = score_net_params.n_head
        ratio = 2 ** (-8 / n_head)
        m = [ratio]
        for i in range(n_head - 1):
            m.append(m[-1] * ratio)
        self.register_buffer("alibi", torch.tensor(m)[None, :, None, None])

    def forward(
        self,
        x: torch.FloatTensor,
        features: dict[str, torch.Tensor],
        r: torch.FloatTensor,
        attention_mask: torch.FloatTensor,
    ) -> torch.Tensor:
        dr = self.point_encoder(r)
        attention_mask = attention_mask[:, None, :, :] * self.alibi
        for index, b in enumerate(self.blocks):
            x = checkpoint(b, x, features, r, dr, attention_mask, use_reentrant=False)

        return x
