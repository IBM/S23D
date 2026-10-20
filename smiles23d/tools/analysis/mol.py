import json
import pickle
from functools import partial
from multiprocessing import Pool

import numpy as np
import py3Dmol
from rdkit import Chem, RDLogger
from rdkit.Chem import AssignStereochemistryFrom3D, SDMolSupplier, rdCIPLabeler
from rdkit.Chem.rdchem import BondType
from rdkit.Geometry import Point3D

RDLogger.DisableLog("rdDistGeom.warning")
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


def graph_to_mol(atoms, bonds, charges, exp_hs):
    # From https://stackoverflow.com/questions/51195392/smiles-from-graph
    # create empty editable mol object
    mol = Chem.RWMol()

    # add atoms to mol and keep track of index
    node_to_idx = [0] * len(atoms)
    for index, (i, hs, charge) in enumerate(zip(atoms, exp_hs, charges)):
        atom = Chem.Atom(i)
        atom.SetNumExplicitHs(hs)
        atom.SetFormalCharge(charge)

        molIdx = mol.AddAtom(atom)
        node_to_idx[index] = molIdx

    # add bonds between adjacent atoms
    for ix, iy, bond_type in bonds:
        mol.AddBond(node_to_idx[ix], node_to_idx[iy], BondType.values[bond_type])

    # # Convert RWMol to Mol object
    mol = mol.GetMol()

    # Update property cache without strict flag
    mol.UpdatePropertyCache(strict=False)

    # Sanitize molecule
    Chem.SanitizeMol(
        mol,
        Chem.SanitizeFlags.SANITIZE_FINDRADICALS
        | Chem.SanitizeFlags.SANITIZE_KEKULIZE
        | Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
        | Chem.SanitizeFlags.SANITIZE_SETCONJUGATION
        | Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION
        | Chem.SanitizeFlags.SANITIZE_SYMMRINGS,
        catchErrors=True,
    )

    return mol, node_to_idx


def show_mol(mol, confId=0, width=200, height=200):
    view = py3Dmol.view(
        data=Chem.MolToMolBlock(
            mol, confId=confId
        ),  # Convert the RDKit molecule for py3Dmol
        style={"stick": {}, "sphere": {"scale": 0.3}},
        width=width,
        height=height,
    )

    view.zoomTo()
    view.show()


def get_chirality(mol):
    AssignStereochemistryFrom3D(mol)
    try:
        rdCIPLabeler.AssignCIPLabels(mol)
    except RuntimeError:
        return []
    labels = []
    for atom in mol.GetAtoms():
        if atom.HasProp("_CIPCode"):
            labels.append(atom.GetProp("_CIPCode"))

    return labels


def create_mol(atoms, charges, edges, explicit_hs, x, multiple_conformers=False):
    def make_one_mol(atoms, charges, edges, explicit_hs, x):
        def remove_hydrogens(atoms, edges, charges, explicit_hs):
            atoms = np.array(atoms)
            charges = np.array(charges)
            explicit_hs = np.array(explicit_hs)
            mask = atoms == "H"
            negative_index = np.arange(len(atoms))[~mask]
            mapping = {j: i for i, j in enumerate(negative_index)}
            index_set = set(np.arange(len(atoms))[mask])
            new_edges = []
            for i in edges:
                if i[0] not in index_set and i[1] not in index_set:
                    new_edges.append([mapping[i[0]], mapping[i[1]], i[2]])
                if i[0] in index_set:
                    explicit_hs[i[1]] += 1
                if i[1] in index_set:
                    explicit_hs[i[0]] += 1
            if not new_edges:
                return atoms.tolist(), edges, charges.tolist(), explicit_hs.tolist()

            return (
                atoms[~mask].tolist(),
                new_edges,
                charges[~mask].tolist(),
                explicit_hs[~mask].tolist(),
            )

        try:
            if len(atoms) != len(x):
                atoms, edges, charges, explicit_hs = remove_hydrogens(
                    atoms, edges, charges, explicit_hs
                )
                add_hs = True
            else:
                add_hs = False

            mol, node_to_idx = graph_to_mol(atoms, edges, charges, explicit_hs)
            conf = Chem.Conformer(mol.GetNumAtoms())
            xi = [x[node_to_idx[i]] for i in range(len(atoms))]
            for index, i in enumerate(xi):
                conf.SetAtomPosition(index, Point3D(i[0], i[1], i[2]))
            mol.AddConformer(conf)
            if add_hs:
                mol = Chem.AddHs(mol, addCoords=True)
            return mol
        except:
            raise

    try:
        if not multiple_conformers:
            atoms, charges, edges, explicit_hs, x = (
                [atoms],
                [charges],
                [edges],
                [explicit_hs],
                [x],
            )

        # Always create reference mol from first conformation, as this is what Generation is based on
        mol = make_one_mol(
            atoms.pop(0), charges.pop(0), edges.pop(0), explicit_hs.pop(0), x.pop(0)
        )
        if mol == None:
            return None

        # Create mol object for each conformation
        mols_for_each_conf = [
            make_one_mol(ai, ci, ei, ehi, xi)
            for ai, ci, ei, ehi, xi in zip(atoms, charges, edges, explicit_hs, x)
        ]
        mols_for_each_conf = [m for m in mols_for_each_conf if m is not None]

        while mols_for_each_conf:
            mol_conf = mols_for_each_conf.pop(0)
            try:
                match = mol_conf.GetSubstructMatch(mol)
                mol_conf = Chem.RenumberAtoms(mol_conf, match)
            except:
                # Don't change atom order if match didn't work....
                # Might cause problems later on TODO: FIX
                mol_conf = mol_conf
            mol.AddConformer(mol_conf.GetConformer(0), assignId=True)

        return mol
    except:
        raise


def create_mol_map(gen_mol, chirality_correction=False):
    molecule = gen_mol["molecule"]
    mol_true = create_mol(
        atoms=[conf["atoms"] for conf in molecule["conformations"]],
        charges=[conf["charges"] for conf in molecule["conformations"]],
        edges=[conf["bonds"] for conf in molecule["conformations"]],
        explicit_hs=[conf["explicitHs"] for conf in molecule["conformations"]],
        x=[
            np.array(conf["position"]).reshape(-1, 3).tolist()
            for conf in molecule["conformations"]
        ],
        multiple_conformers=True,
    )
    if chirality_correction:
        true_chirality = get_chirality(mol_true)
    mol_pred_list = []
    for prediction in gen_mol["predictions"]:
        mol = create_mol(
            atoms=molecule["conformations"][0]["atoms"],
            charges=molecule["conformations"][0]["charges"],
            edges=molecule["conformations"][0]["bonds"],
            explicit_hs=molecule["conformations"][0]["explicitHs"],
            x=prediction,
        )
        if chirality_correction:
            pred_chirality = get_chirality(mol)
            if true_chirality != pred_chirality:
                pos = mol.GetConformer().GetPositions()
                pos[:, 2] *= -1
                mol.GetConformer().SetPositions(pos)
        mol_pred_list.append(mol)
    return (mol_pred_list, mol_true, molecule["smiles"])


class MolData:
    """
    Functions are based on 'molecule' objects that have the structure
    molecule = {
        "smiles": <smiles>,
        "atoms": [<atom1>, <atom2>, ...],
        "bonds": [(<node1>, <node2>, <bondType>), (...), ...]
        "positions": [
            [<x1>, <x2>, ...] <- 1D list 3D positions of atoms
        ], <- 2D list of all conformations in ground truth data
        "predictions": [
            [[<x1>, <x2>, <x3>], [...], ...] <- 2D list of 3D positions of atoms
        ] <- 3D list of all generated conformations
    }.

    MolData object
    """

    def __init__(
        self,
        filenames=None,
        n_cpus=1,
        n_files=1,
        chirality_correction: bool = False,
    ):
        self.filenames = filenames
        self.n_cpus = n_cpus
        self.chirality_correction = chirality_correction
        self.mol_data = []  # list of tuples (mol_pred, mol_true, smiles)
        self.mol_pred = (
            []
        )  # list of lists of Mol objects that each have one predicted conformation
        self.mol_true = []  # List of Mol objects that contain multiple conformations
        self.smiles = []  # List of smiles strings
        if self.filenames:
            self.load_molecules_jsonl_file(self.filenames, n_files=n_files)

    def load_jsonl_files(self, filenames, n_files=1):
        if isinstance(filenames, str):
            filenames = [filenames]

        mol_list = []
        for base_filename in filenames:
            for i in range(n_files):
                filename = f"{base_filename.removesuffix('.jsonl')}_{i}.jsonl"
                with open(filename) as f:
                    for mol in f:
                        mol_list.append(json.loads(mol))

        return mol_list

    def load_molecules_jsonl_file(self, filenames=None, n_files=1):
        mol_list = self.load_jsonl_files(filenames=filenames, n_files=n_files)
        self.load_molecules(mol_list=mol_list)

    def load_molecules(self, mol_list):
        create_mol_map_parallel = partial(
            create_mol_map, chirality_correction=self.chirality_correction
        )

        with Pool(self.n_cpus) as p:
            mol_data_list = p.map(create_mol_map_parallel, mol_list)

        # transpose list of tuples
        mol_data = list(map(list, zip(*mol_data_list)))

        # filter predicted cells that failed to create Mol object
        mol_dict = {}
        for mol_pred, mol_true, smiles in zip(*mol_data):
            if smiles not in mol_dict.keys():
                mol_dict[smiles] = [None, None, smiles]

            if mol_true is not None:
                if mol_dict[smiles][1] == None:
                    mol_dict[smiles][1] = mol_true

            if len(mol_pred) > 0:
                if mol_dict[smiles][0] == None:
                    mol_dict[smiles][0] = mol_pred
                else:
                    mol_dict[smiles][0].extend(mol_pred)

        mol_data = list(mol_dict.values())

        self.mol_data = mol_data
        mapped_md = list(map(list, zip(*self.mol_data)))
        self.mol_pred, self.mol_true, self.smiles = (
            mapped_md[0],
            mapped_md[1],
            mapped_md[2],
        )

    def load_sdf_files(self, pred_file=None, true_file=None, cond_file=None):
        def load_sdf(filename):
            ms = []
            with SDMolSupplier(filename) as r:
                for mol in r:
                    ms.append(mol)
            return ms

        if pred_file:
            self.mol_pred = load_sdf(pred_file)
        if true_file:
            self.mol_true = load_sdf(true_file)
        if cond_file:
            self.mol_cond = load_sdf(cond_file)

    def to_pickle(self, filename: str):
        pickle.dump(self.mol_data, open(filename, "wb"))

    def __call__(self, idx):
        record = {
            "mol_pred": self.mol_data[idx][0],
            "mol_true": self.mol_data[idx][1],
            "smiles": self.smiles[idx],
        }
        return record

    def __len__(self):
        return len(self.mol_data)
