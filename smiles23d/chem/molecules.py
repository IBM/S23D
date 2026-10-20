import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


def graph_to_adj_list(n_atoms, edges):
    """
    Convert list of (begin_atom_idx, end_atom_idx, bond_type) to adjacency list.

    Args:
    ----
        n_atoms (int): Number of atoms.
        edges( tuple[int]): (begin_atom_idx, end_atom_idx, bond_type).
    """
    vertices = [[] for i in range(n_atoms)]
    for edge in edges:
        vertices[edge[0]].extend([edge[1], edge[2]])
    flat = []
    for index, i in enumerate(vertices):
        flat.append(index)
        flat.extend(i)
        flat.append(-1)
    flat.append(n_atoms)
    flat = np.array(flat, dtype=np.int32)
    return flat


def mol_to_smiles23d(conformation):
    edges = [
        (
            bond.GetBeginAtomIdx(),
            bond.GetEndAtomIdx(),
            int(bond.GetBondType()),
        )
        for bond in conformation.GetBonds()
    ]

    atoms = list(conformation.GetAtoms())
    exp_hs = [atom.GetNumExplicitHs() for atom in atoms]
    charges = [atom.GetFormalCharge() for atom in atoms]
    atoms = [atom.GetSymbol() for atom in atoms]

    positions = conformation.GetConformer().GetPositions()

    graph_adj_list = graph_to_adj_list(len(atoms), edges)
    mean = np.mean(positions, axis=0, keepdims=True)
    positions = (positions - mean).flatten().astype(np.float64)

    molecule = {
        "atoms": atoms,
        "graph": graph_adj_list,
        "explicitHs": exp_hs,
        "charges": charges,
        "positions": positions,
    }
    return molecule


def check_and_add_hs_to_mol(mol: Chem.Mol):
    mol_has_implicit_hs = False
    for atom in mol.GetAtoms():
        if atom.GetTotalNumHs() > 0:
            mol_has_implicit_hs = True
            break

    if mol_has_implicit_hs:
        try:
            mol = Chem.AddHs(mol, addCoords=True)
        except Exception:
            raise ValueError(f"Could not AddHs for: {mol}")

    return mol


def embed_rdkit_conformer(mol: Chem.Mol, max_iter=10):
    if mol.GetNumConformers() == 0:
        try:
            for seed in range(max_iter):
                params = AllChem.ETKDGv3()
                params.randomSeed = seed
                params.embedFragmentsSeparately = False
                params.useRandomCoords = True
                embed_result = AllChem.EmbedMolecule(mol, params)
                if embed_result > -1:
                    break
        except Exception:
            raise ValueError(f"Could not embed a conformer for: {mol}")
        if mol.GetNumConformers() == 0:
            raise ValueError(
                f"Could not embed a conformer for: {Chem.MolToSmiles(mol)}"
            )

    return mol
