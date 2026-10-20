import numpy as np

_MASK_TOKEN = 1

_vocab = {
    "PAD": 0,
    "MASK": _MASK_TOKEN,
    "BRICKS": 2,
    "B": 3,
    "Bi": 4,
    "Br": 5,
    "C": 6,
    "Cl": 7,
    "F": 8,
    "H": 9,
    "I": 10,
    "N": 11,
    "O": 12,
    "P": 13,
    "S": 14,
    "Si": 15,
}


class PileTokenizer:
    def __init__(self):
        self._n_tokens = len(_vocab)

    def tokenize_atoms(self, atoms):
        return np.array([_vocab.get(i, _MASK_TOKEN) for i in atoms], dtype=np.int64)

    @property
    def n_tokens(self):
        return self._n_tokens

    @property
    def vocab(self):
        return _vocab

    @property
    def reverse_vocab(self):
        return {value: key for key, value in _vocab.items()}
