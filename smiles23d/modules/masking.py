import torch


def create_masks(x, n_aux_tokens, tensor=None):
    """Create boolean mask for attention to ignore padding tokens and inf mask for equiv network."""
    B, T = x.size()
    if tensor is not None:
        tensor = torch.cat(
            [torch.zeros(B, n_aux_tokens, T, device=x.device), tensor],
            dim=1,
        )
        tensor = torch.cat(
            [
                torch.zeros(B, T + n_aux_tokens, n_aux_tokens, device=x.device),
                tensor,
            ],
            dim=-1,
        )

    boolean_mask = x == 0
    padding_mask = torch.ones_like(boolean_mask, dtype=torch.float32)
    padding_mask.masked_fill_(boolean_mask, 0.0)

    # Attention mask should handle aux tokens for time variable
    aux_tokens = torch.ones(B, n_aux_tokens, dtype=x.dtype, device=x.device)
    x = torch.cat([aux_tokens, x], dim=-1)
    boolean_mask = x == 0

    attention_mask = torch.zeros(*x.shape, x.shape[-1], device=x.device)
    attention_mask.masked_fill_(boolean_mask.unsqueeze(dim=-1), float("-inf"))
    attention_mask.masked_fill_(boolean_mask.unsqueeze(dim=-2), float("-inf"))

    if tensor is not None:
        attention_mask = tensor + attention_mask

    return attention_mask, attention_mask[:, n_aux_tokens:, n_aux_tokens:], padding_mask


def create_graph_encoder_masks(x):
    """Create boolean mask for attention to ignore padding tokens and inf mask for equiv network."""
    B, T = x.size()
    boolean_mask = x == 0
    padding_mask = torch.ones_like(boolean_mask, dtype=torch.float32)
    padding_mask.masked_fill_(boolean_mask, 0.0)

    attention_mask = torch.ones(
        *x.shape, x.shape[-1], dtype=torch.bool, device=x.device
    )
    attention_mask.masked_fill_(boolean_mask.unsqueeze(dim=-1), False)
    attention_mask.masked_fill_(boolean_mask.unsqueeze(dim=-2), False)

    return attention_mask, padding_mask


def column_permuation(x, permutation):
    """Does permutation based on permutation (1-based permutation indices)."""
    B, T = permutation.size()
    C = x.shape[2]
    template = torch.zeros(B, T, C, device=x.device, dtype=x.dtype)
    for i, perm in enumerate(permutation - 1):
        template[i] = x[i, perm]
    return template
