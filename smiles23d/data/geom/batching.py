from collections import defaultdict
from types import SimpleNamespace

import torch


def _pad(tensor, n_pad):
    """
    Pad a tensor with zeros.

    Args:
    ----
        tensor(torch.Tensor): Tensor to pad.
        n_pad(int): Number of padding elements to add.

    Returns:
    -------
        torch.Tensor: Padded Tensor
    """
    if n_pad == 0:
        return tensor
    zero = torch.zeros_like(tensor[0])[None, ...]
    padded_tensor = torch.cat([tensor] + [zero] * n_pad, dim=0)
    return padded_tensor


class FeatureBatch:
    """
    Building of batch of features, with padding, and setting them into
        a SimpleNamespace object.

    Methods
    -------
        add(dict[features in the batch]): Adds features into the batch.
        get_batch: Returns a batch of features with padding.
    """

    def __init__(self):
        self.batch = defaultdict(list)

    def add(self, **kargs):
        for key, values in kargs.items():
            self.batch[key].append(values)

    def get_batch(self):
        first_key = next(iter(self.batch.values()))
        n_max = max([len(i) for i in first_key])
        n_pads = [n_max - len(i) for i in first_key]
        padded_batch = {}
        for key, values in self.batch.items():
            padded_batch[key] = torch.cat(
                [_pad(v, n_p)[None, ...] for v, n_p in zip(values, n_pads)], dim=0
            )

        return SimpleNamespace(**padded_batch)
