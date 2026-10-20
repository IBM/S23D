from dataclasses import dataclass


@dataclass
class ScoreNetParams:
    layers: list[int]
    n_embd: int
    n_head: int
    vocab_size: int
    n_aux_tokens: int
    n_time_basis: int
    translational_augmentation: float


@dataclass
class GraphEncoderParams:
    use_degree: bool
    use_charges: bool
    use_bonds: bool
