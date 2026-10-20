"""
RMSD calculations between predicted and ground-truth conformations.

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
}

MolData objects from tools.mol have this structure:
    self.mol_data = [] # list of tuples (mol_pred, mol_true, smiles)
    self.mol_pred = [] # list of lists of Mol objects that each have one predicted conformation
    self.mol_true = [] # List of Mol objects that contain multiple conformations
    self.mol_cond = [] # List of Mol objects that each have one conformation ???
    self.smiles = [] # List of smiles strings
The idea is that Posebusters functions generally use a predicted Mol object with a
single conformation, and may compare against a true Mol object with multiple conformations
"""

import time
from functools import partial

# from itertools import batched # Requires python 3.12
from itertools import islice
from multiprocessing import Pool

import numpy as np
from posebusters.modules.rmsd import robust_rmsd
from rdkit import Chem

from ...tools import MolData


def batched(iterable, n, *, strict=False):
    if n < 1:
        raise ValueError("n must be at least one")
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        if strict and len(batch) != n:
            raise ValueError("batched(): incomplete batch")
        yield batch


def rmsds_batch(
    mol_pred_list,
    mol_true_list,
    conf_id_list,
    fill_value=np.nan,
    heavy_only=True,
    kabsch=True,
):
    rmsds = []
    for mol_pred, mol_true, conf_id in zip(mol_pred_list, mol_true_list, conf_id_list):
        try:
            rmsds.append(
                robust_rmsd(
                    mol_probe=mol_pred,
                    mol_ref=mol_true,
                    conf_id_ref=conf_id,
                    heavy_only=heavy_only,
                    kabsch=kabsch,
                )
            )
        except:
            rmsds.append(fill_value)
    return rmsds


def calc_rmsds_batches(
    mol_pred_list: list,
    mol_true_list: list,
    conf_id_list: list | None = None,
    rmsds_batch_size: int = 1,
    kabsch: bool = True,
    heavy_only: bool = True,
    n_cpus: int | None = None,
    fill_value=np.nan,
):
    if not conf_id_list:
        conf_id_list = [0] * len(mol_pred_list)

    batches = zip(
        batched(mol_pred_list, rmsds_batch_size),
        batched(mol_true_list, rmsds_batch_size),
        batched(conf_id_list, rmsds_batch_size),
    )

    t0 = time.time()

    with Pool(n_cpus) as p:
        rmsd_results_batched = p.starmap(
            partial(
                rmsds_batch,
                fill_value=fill_value,
                heavy_only=heavy_only,
                kabsch=kabsch,
            ),
            batches,
        )

    rmsd_results = []
    list(map(rmsd_results.extend, rmsd_results_batched))

    return rmsd_results


def calc_all_rmsds(
    mol_data: MolData, kabsch=True, heavy_only=True, n_cpus=1, rmsds_batch_size=None
):
    mol_dat = mol_data.mol_data

    t0 = time.time()
    # Create flattened list of all RMSD calculations to carry out
    mol_pred_list = []
    mol_true_list = []
    conf_id_list = []
    indices_list = []
    rmsds_list = []
    rmsd_mat_list = []
    for i, (mol_preds, mol_true, _) in enumerate(mol_dat):
        try:
            n_conf = mol_true.GetNumConformers()
        except:
            n_conf = 0
        rmsd_mat_list.append(
            np.array((len(mol_preds), n_conf, 2))  # shape (n_mol_pred, n_conf, 2)
        )
        for j in range(n_conf):
            try:
                mt = Chem.Mol(mol_true, confId=j)
            except:
                mt = None
            if mt and mol_preds:
                for k, mp in enumerate(mol_preds):
                    if mp:
                        mol_pred_list.append(mp)
                        mol_true_list.append(mt)
                        conf_id_list.append(j)
                        indices_list.append((i, j, k))
            else:
                mol_pred_list.append(None)
                mol_true_list.append(None)
                conf_id_list.append(j)
                indices_list.append((i, j, 0))

    if not rmsds_batch_size:
        rmsds_batch_size = max(1, -(-len(mol_pred_list) // (n_cpus * 16)))

    t1 = time.time()
    print(f"time to flatten RMSDs: {t1-t0}")
    print(
        f"n rmsd calcs: {len(mol_pred_list)} -- rmsd batch size: {rmsds_batch_size} -- rmsd batches: {len(mol_pred_list)/rmsds_batch_size}"
    )

    t0 = time.time()
    # Divide RMSD calculations into even sized batches and run them on ray cluster
    rmsds_list = calc_rmsds_batches(
        mol_pred_list=mol_pred_list,
        mol_true_list=mol_true_list,
        conf_id_list=conf_id_list,
        rmsds_batch_size=rmsds_batch_size,
        kabsch=kabsch,
        heavy_only=heavy_only,
        n_cpus=n_cpus,
    )
    t1 = time.time()
    print(f"time to calc rmsds: {t1-t0}")

    t0 = time.time()
    # Populate RMSD matrices
    rmsd_mat_list = []
    for mp, mt, _ in mol_dat:
        if mt:
            rmsd_mat_list.append(np.ones((len(mp), mt.GetNumConformers())) * np.nan)
        else:
            rmsd_mat_list.append(np.ones((len(mp), 1)) * np.nan)
    for rmsd, ind in zip(rmsds_list, indices_list):
        (i, j, k) = ind
        rmsd_mat_list[i][k, j] = rmsd

    t1 = time.time()
    print(f"time to populate rmsd matrices: {t1-t0}")

    return rmsd_mat_list


def rmsd_recall_COV(all_mol_rmsds: list, threshold: float = 1.0):
    # Parameter all_mol_rmsds obeying format from calc_all_rmsds function
    # Recall-based coverage score algorithm:
    # i) For each True conformation (R):
    #    ii) is there at least one predicted conformation (R^) that is within a
    #        threshold (δ) of R in rmsd?
    # iii) Count the fraction of true conformations from all true molecules that are
    #      "covered" (i.e. for which the criteria above passes)
    cov_per_mol = [np.mean(np.nanmin(m, axis=0) <= threshold) for m in all_mol_rmsds]
    return np.mean(cov_per_mol), np.median(cov_per_mol)


def rmsd_precision_COV(all_mol_rmsds: list, threshold: float = 1.0):
    # Parameter all_mol_rmsds obeying format from calc_all_rmsds function
    # Precision-based coverage score algorithm:
    # i) For each Generated conformation (R^):
    #    ii) is there at least one true conformation (R) that is within a
    #        threshold (δ) of R^ in rmsd?
    # iii) Count the fraction of generated conformations from all molecules that are
    #      "covered" (i.e. for which the criteria above passes)
    cov_per_mol = [np.mean(np.nanmin(m, axis=1) <= threshold) for m in all_mol_rmsds]
    return np.mean(cov_per_mol), np.median(cov_per_mol)


def rmsd_recall_MAT(all_mol_rmsds: list):
    # Recall-based matching score algorithm
    # i) Get min rmsd between R and R^ for each R (true conformation))
    # ii) compute mean mins across all true molecules and conformations
    mat_per_mol = [np.nanmin(m, axis=0).mean() for m in all_mol_rmsds]
    return (np.nanmean(mat_per_mol)), np.nanmedian(mat_per_mol)


def rmsd_precision_MAT(all_mol_rmsds: list):
    # Recall-based matching score algorithm
    # i) Get min rmsd between R^ and R for each R^ (generated conformation)
    # ii) compute mean mins across all generated molecules and conformations
    mat_per_mol = [np.nanmin(m, axis=1).mean() for m in all_mol_rmsds]
    return (np.nanmean(mat_per_mol)), np.nanmedian(mat_per_mol)


class RMSDWrapper:
    """Class to run RMSD on a set of molecule data."""

    def __init__(
        self,
        mol_data: MolData,
        align=True,
        heavy_only=True,
        n_cpus=1,
        return_metrics=True,
        return_rmsds=False,
        threshold=0.75,
    ):
        self.mol_data = mol_data
        self.kabsch = align
        self.heavy_only = heavy_only
        self.n_cpus = n_cpus
        self.return_metrics = return_metrics
        self.return_rmsds = return_rmsds
        self.threshold = threshold

    def __call__(self):
        all_mol_rmsds = calc_all_rmsds(
            mol_data=self.mol_data,
            kabsch=self.kabsch,
            heavy_only=self.heavy_only,
            n_cpus=self.n_cpus,
        )
        recall_cov_mean, recall_cov_median = rmsd_recall_COV(
            all_mol_rmsds, threshold=self.threshold
        )
        precision_cov_mean, precision_cov_median = rmsd_precision_COV(
            all_mol_rmsds, threshold=self.threshold
        )
        recall_mat_mean, recall_mat_median = rmsd_recall_MAT(all_mol_rmsds)
        precision_mat_mean, precision_mat_median = rmsd_precision_MAT(all_mol_rmsds)
        if self.return_metrics:
            return_dict = {
                "CovR_mean": recall_cov_mean,
                "CovR_med": recall_cov_median,
                "CovP_mean": precision_cov_mean,
                "CovP_med": precision_cov_median,
                "MatR_mean": recall_mat_mean,
                "MatR_med": recall_mat_median,
                "MatP_mean": precision_mat_mean,
                "MatP_med": precision_mat_median,
            }
        else:
            return_dict = {}
        if self.return_rmsds:
            return_dict["all_mol_rmsds"] = all_mol_rmsds
        return return_dict
