import time
from collections import Counter
from collections.abc import Sequence

import numpy as np
import torch
from numpy.random import Generator
from numpy.typing import NDArray
from scipy.stats import ortho_group, special_ortho_group

from smiles23d.nexusnet.distributed import get_rank
from smiles23d.tokenizers.piletok import PileTokenizer

from .alibi import get_shortest_path, pad_paths
from .batching import FeatureBatch


class NotToUnwrap:
    """Preventing PL to unwrap the dictionary and lists."""

    def __init__(self, value):
        self.value = value


def rotation_transform(rng: Generator, x: NDArray, use_so3: bool):
    """
    Orthogonal transform / rotation augmentation.

    Args:
    ----
        x: array of atom coordinates
        use_so3: Use rotations instead of orthogonal transformations.
    """
    if use_so3:
        rot_mat = special_ortho_group.rvs(dim=3, random_state=rng)
    else:
        rot_mat = ortho_group.rvs(dim=3, random_state=rng)
    return x @ rot_mat


class GEOMCollator:
    """
    Collator for the GEOM dataset.

    Args:
    ----
        tokenizer (PileTokenizer): Tokenizer to use.
        num_workers (int): Number of workers.
        samples_per_molecule (int): The number of samples to collate to
            the batch for each molecule in the batch. Default: 1.
        removeHs (bool): Whether to remove hydrogen atoms when
            collating the data into the batch. Default: False.
        use_so3 (bool): Whether to use SO(3) rotations. Default: False.

    Returns:
    -------
        A dictionary containing the features for the batch. Format:
        features = {
            "x": batch.x,
            "atoms": batch.atoms,
            "charges": batch.charges,
            "short_paths": short_paths,
            "bonds": batch.bond_type,
            "degree": batch.degree[..., None],
        }
    """

    conformation_field = "conformation"
    conformations_field = "conformations"
    n_edge_features = 3
    bond_map = {12: 0, 2: 1, 3: 2}

    def __init__(
        self,
        tokenizer: PileTokenizer,
        num_workers: int,
        samples_per_molecule: int = 1,
        removeHs: bool = False,
        use_so3: bool = False,
        **kargs,
    ):
        self.tokenizer = tokenizer
        self.samples_per_molecule = samples_per_molecule
        self.augmentation = False
        self.num_workers = num_workers
        self.removeHs = removeHs
        self.use_so3 = use_so3

    def setup_rng(self, id: int):
        """
        Initializes RNG with different seeds for different ranks and
            workers. The seed is rank * num_workers + id (or the rank when
            num_workers is 0) plus the wall-clock time in hundredths of a
            second, so the RNG is not reproducible between runs.

        Args:
        ----
            id (int): Worker id added to the seed when num_workers is
                not 0.
        """
        rank = get_rank()
        current_time = int(time.time() * 100)
        seed = rank * self.num_workers + id if self.num_workers else rank
        self.rng = np.random.default_rng(seed + current_time)

    def setup_augmentation(
        self,
        mask_prob: float | None,
    ):
        """
        Set probability to apply atom masking to molecules.

        Args:
        ----
            mask_prob (float, optional): Masking probability
        """
        if mask_prob is None:
            self.augmentation = False
        else:
            self.augmentation = True
            self.mask_prob = 0.0 if mask_prob is None else mask_prob

    def get_augmentation_bits(self, x):
        """
        Get the augmentation bits for a batch.

        Args:
        ----
            x (list): Original data

        Returns: array that contains 0,1
            0 - no augmentation for an item in the batch
            1 - mask augmentation.
        """
        n = len(x)
        if not self.augmentation:
            return [0] * n

        return self.rng.choice(
            a=2,
            size=n,
            p=[1 - self.mask_prob, self.mask_prob],
            replace=True,
        )

    @torch.no_grad
    def __call__(self, x):
        feature_batch = FeatureBatch()
        short_paths = []
        augmentation_bits = self.get_augmentation_bits(x)

        for i, augmentation_bit in zip(x, augmentation_bits):
            conformation = i["conformation"]
            atoms, charges, edges, positions = (
                conformation["atoms"],
                conformation["charges"],
                conformation["graph"],
                conformation["positions"],
            )

            positions = np.copy(positions.reshape(-1, 3))
            edges = list2bonds(edges)
            if self.removeHs:
                atoms, edges, charges, positions = remove_hydrogens(
                    atoms, edges, charges, positions
                )

            positions -= np.mean(positions, axis=0)
            positions = rotation_transform(self.rng, positions, self.use_so3)

            if augmentation_bit == 1:
                atoms = mask_atoms(self.rng, atoms)

            bond_type = torch.zeros(len(atoms), GEOMCollator.n_edge_features)
            for b1, b2, b in edges:
                edge_feature_index = self.bond_map.get(b)
                bond_type[b1, edge_feature_index] = 1
                bond_type[b2, edge_feature_index] = 1

            edges = torch.tensor(edges, dtype=torch.int64)[:, :2]
            degree = torch.zeros(len(atoms))
            for i, j in edges:
                degree[i] += 1
                degree[j] += 1

            short_path = get_shortest_path(edges.tolist())
            short_paths.append(short_path)

            atoms = self.tokenizer.tokenize_atoms(atoms)
            atoms = torch.from_numpy(atoms.astype(np.int64))

            positions = torch.from_numpy(positions.astype(np.float32))
            feature_batch.add(
                x=positions,
                charges=tokenize_charges(charges),
                atoms=atoms,
                bond_type=bond_type,
                degree=degree,
            )

        batch = feature_batch.get_batch()
        short_paths = torch.from_numpy(pad_paths(short_paths))
        features = {
            "x": batch.x,
            "atoms": batch.atoms,
            "charges": batch.charges,
            "short_paths": short_paths,
            "bonds": batch.bond_type,
            "degree": batch.degree[..., None],
        }
        return features

    def predict_collator(self, batch):
        """
        Collator function for predict and test. Uses
            samples_per_molecule to duplicate data items into an
            'inference_batch', which then gets collated as normal.
            Stores the full list of 'true' conformers for each molecule
            to the 'molecules' field of the output.

        Args:
        ----
            batch (list): Data items to be processed into a batch.

        Returns:
        -------
            tuple: A tuple containing the features and molecules.
        """
        inference_batch = []
        molecules = []
        for item in batch:
            conformations = item.get(GEOMCollator.conformations_field, None)
            conformations_field = []
            if conformations:
                for _ in range(self.samples_per_molecule):
                    inference_batch.append(
                        {GEOMCollator.conformation_field: conformations[0]}
                    )
                for conformation in conformations:
                    conformation_field = {
                        "position": conformation["positions"].tolist(),
                        "atoms": conformation["atoms"],
                        "explicitHs": conformation["explicitHs"],
                        "charges": conformation["charges"],
                        "bonds": list2bonds(conformation["graph"]),
                    }
                    conformations_field.append(conformation_field)

            molecule = {"smiles": item["smiles"], "conformations": conformations_field}
            molecules.append(molecule)
        features = self(inference_batch)
        return features, NotToUnwrap(molecules)


def list2bonds(array: NDArray[np.int_]):
    """
    Creates a list of bonds for each atom.

    Args:
    ----
        array (NDArray[np.int_]): Array of atoms to compute bonds for.

    Returns:
    -------
        edges (list): List of bonds for each atom.
    """
    n = array[-1]
    bounds = np.insert(np.where(array == -1)[0], 0, -1)
    edges = []
    for start_idx, end_idx in zip(bounds[:-1], bounds[1:]):
        node_idx = array[start_idx + 1]
        for i in range(start_idx + 2, end_idx, 2):
            edges.append([node_idx.item(), array[i].item(), array[i + 1].item()])
    return edges


def mask_atoms(
    rng: Generator, atoms: Sequence, ratio: float = 0.1, mask_symbol: str = "MASK"
):
    """
    Masks at most one atom, chosen uniformly at random.

    The number of masked atoms is min(round(len(atoms) * ratio), 1). With the
    default ratio of 0.1, a molecule with 6 or more atoms has exactly one atom
    masked and a molecule with 5 or fewer atoms has none. Every atom has the
    same probability of being chosen. The atoms are replaced in place.

    Args:
    ----
        rng (Generator): A random number generator for selecting atoms
            to mask.
        atoms (Sequence): A mutable sequence of atom symbols.
        ratio (float, optional): Fraction of atoms that is rounded to give
            the number of atoms to mask, capped at one. Defaults to 0.1.
        mask_symbol (str, optional): The symbol to replace masked atoms
            with. Defaults to "MASK".

    Returns:
    -------
        atoms (Sequence): The same sequence, with any masked atom replaced
            by `mask_symbol`.
    """
    n = len(atoms)
    m = min(round(n * ratio), 1)
    atom_voc = Counter(atoms)
    total = sum(atom_voc.values())
    atom_probs = {key: 1.0 / total for key, value in atom_voc.items()}
    probs = [atom_probs[i] for i in atoms]
    mask_index = rng.choice(n, size=m, replace=False, p=probs)
    for i in mask_index:
        atoms[i] = mask_symbol
    return atoms


def tokenize_charges(charges: NDArray[np.int_]):
    """
    One-hot encoding of charge categories [negative, zero, positive].

    Args:
    ----
        charges (NDArray[np.int_]): A numpy array of integer charges.

    Returns:
    -------
        torch.Tensor: A one-hot encoded tensor of shape
            (len(charges), 3).
    """
    charges = np.sign(charges) + 1
    values = [[0, 1], [0, 0], [1, 0]]
    charges = [values[i] for i in charges]
    return torch.tensor(charges, dtype=torch.float32)


def remove_hydrogens(atoms, edges, charges, positions):
    """
    Removes hydrogen atoms from a molecule and adjusts all features
        (e.g., removes edges with hydrogen, slices hydrogen positions
        out of positions, etc.).

    Args:
    ----
        atoms (list): Atom symbols.
        edges (list): Edges in the molecule.
        charges (list): Charges for each atom.
        positions (NDArray[np.float32]): Positions for each atom.

    Returns:
    -------
        tuple: Updated atoms, edges, charges, and positions, with
            hydrogens removed.
    """
    atoms = np.array(atoms)
    charges = np.array(charges)
    mask = atoms == "H"
    negative_index = np.arange(len(atoms))[~mask]
    mapping = {j: i for i, j in enumerate(negative_index)}
    index_set = set(np.arange(len(atoms))[mask])
    new_edges = [
        [mapping[i[0]], mapping[i[1]], i[2]]
        for i in edges
        if i[0] not in index_set and i[1] not in index_set
    ]
    if not new_edges:
        return atoms, edges, charges, positions

    return atoms[~mask], new_edges, charges[~mask], positions[~mask]
