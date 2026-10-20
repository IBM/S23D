#!/usr/bin/env python

"""See README.md for additional description."""

import argparse
import glob
import os
import pickle
import shutil
import time

import igraph
import numpy as np
from rdkit import Chem

from smiles23d.nexusdb.datasets.nexus import build_litdata_dataset_index
from smiles23d.nexusdb.shuffle import (
    DictionarySchema,
    ListSchema,
    NumpySchema,
    ObjectSchema,
    StrSchema,
    build_dataset_multiprocess,
    group_dataset,
    shuffle_collection,
    shuffle_each_collection_file,
    sort_collection,
    sort_each_collection_file,
    unroll_schema,
)
from smiles23d.nexusdb.shuffle.filters import filter_dataset


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


def is_connected(atoms, edges):
    igraph_edges = [[i, j] for i, j, b in edges]
    g = igraph.Graph(n=len(atoms), edges=igraph_edges)
    return g.is_connected()


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
    if not is_connected(atoms, edges):
        return None
    canonical_smiles = Chem.MolToSmiles(conformation, canonical=True)
    exp_hs = [atom.GetNumExplicitHs() for atom in atoms]
    charges = [atom.GetFormalCharge() for atom in atoms]
    atoms = [atom.GetSymbol() for atom in atoms]

    positions = conformation.GetConformer().GetPositions()

    graph_adj_list = graph_to_adj_list(len(atoms), edges)
    mean = np.mean(positions, axis=0, keepdims=True)
    positions = (positions - mean).flatten().astype(np.float64)

    molecule = {
        "smiles": canonical_smiles,
        "atoms": atoms,
        "graph": graph_adj_list,
        "explicitHs": exp_hs,
        "charges": charges,
        "positions": positions,
    }
    return molecule


def pkl2mol(filename):
    """
    Python generator that convert GEOM record in the Wang2024 (MCF model) format
    to output dictionaries for every SMILES.

    A record in form of dictionary contains SMILES string under "smi", an rdkit mol
    object under 'mol', atoms positions under 'pos' and eigenvalues and eigenvectors
    under 'eig_vals' and 'eig_vecs'
    """
    filename, dataset_name = filename
    with open(filename, "rb") as file:
        obj = pickle.load(file)

    molecule = mol_to_smiles23d(obj["mol"])
    if molecule and len(molecule["atoms"]) > 1:
        molecule["dataset"] = dataset_name
        molecule["smiles"] = obj["smi"]
        molecule["eig_vals"] = obj["eig_vals"].numpy()
        molecule["eig_vecs"] = obj["eig_vecs"].numpy()
        yield molecule


def get_geom_filenames(root, max_confs=20):
    train_filenames = [
        (i, "drugs") for i in sorted(glob.glob(os.path.join(root, "train/*.pkl")))
    ]
    train_filenames = [
        fn
        for fn in train_filenames
        if int(fn[0].split(".")[-2].split("_")[-1]) < max_confs
    ]
    val_filenames = [
        (i, "drugs") for i in sorted(glob.glob(os.path.join(root, "val/*.pkl")))
    ]
    val_filenames = [
        fn
        for fn in val_filenames
        if int(fn[0].split(".")[-2].split("_")[-1]) < max_confs
    ]
    test_filenames = [
        (i, "drugs") for i in sorted(glob.glob(os.path.join(root, "test/*.pkl")))
    ]
    test_filenames = [
        fn for fn in test_filenames if int(fn[0].split(".")[-2].split("_")[-1]) < 10000
    ]
    test_1000_filenames = [
        (i, "drugs") for i in sorted(glob.glob(os.path.join(root, "test_1000/*.pkl")))
    ]
    test_1000_filenames = [
        fn
        for fn in test_1000_filenames
        if int(fn[0].split(".")[-2].split("_")[-1]) < 10000
    ]
    return (train_filenames, val_filenames, test_filenames, test_1000_filenames)


def build_sort_and_group_dataset(
    filenames, split, schema, group_schema, freq: float = 1.0
):
    # Convert pickle files into a raw dataset in Nexus format
    # Every item is SMILE and has multiple conformations

    start = time.time()
    build_dataset_multiprocess(
        data_generator=pkl2mol,
        generator_inputs=filenames,
        schema=schema,
        output_dir=prefix_filename(f"data_{split}"),
        chunk_size=100000,
        n_processes=10,
        queue_chunk_size=10000,
    )
    end = time.time()
    print(f"Build geom drugs {split} execution time {end - start}s.")

    # Sort by SMILES
    start = time.time()
    sort_each_collection_file(
        input_dir=prefix_filename(f"data_{split}"),
        output_dir=prefix_filename(f"sorted_file_{split}"),
        schema=schema,
        key=lambda x: x["smiles"],
    )
    end = time.time()
    print(f"Sort files {split} execution time {end - start}s.")

    start = time.time()
    sort_collection(
        input_dir=prefix_filename(f"sorted_file_{split}"),
        output_dir=prefix_filename(f"sorted_{split}"),
        chunk_size=100000,
        schema=schema,
        key=lambda x: x["smiles"],
    )
    end = time.time()
    shutil.rmtree(prefix_filename(f"sorted_file_{split}"))
    shutil.rmtree(prefix_filename(f"data_{split}"))
    print(f"Sort collection {split} execution time {end - start}s.")

    start = time.time()
    group_dataset(
        key="smiles",
        new_key="conformations",
        schema=schema,
        new_schema=group_schema,
        input_dir=prefix_filename(f"sorted_{split}"),
        output_dir=prefix_filename(f"grouped_prefilter{split}"),
        chunk_size=5000,
    )
    end = time.time()
    shutil.rmtree(prefix_filename(f"sorted_{split}"))
    print(f"Build grouped pile execution time {end - start}s.")

    # Filter SMILES down to approx. freq
    start = time.time()

    print(f"Filtering: downsample to {freq}")

    def filter_by_freq(item):
        return np.random.uniform() < freq

    filter_dataset(
        input_dir=prefix_filename(f"grouped_prefilter{split}"),
        output_dir=prefix_filename(f"grouped_{split}"),
        schema=group_schema,
        filter=filter_by_freq,
        chunk_size=5000,
    )
    end = time.time()
    shutil.rmtree(prefix_filename(f"grouped_prefilter{split}"))
    print(f"Filter grouped pile execution time {end - start}s.")


def sort_grouped_dataset(split, group_schema):
    # Sort split based on number of atoms
    # Similar to shuffling, first we sort each binary file
    # and merge files after.
    start = time.time()
    sort_each_collection_file(
        input_dir=prefix_filename(f"grouped_{split}"),
        output_dir=prefix_filename(f"sorted_{split}_files"),
        schema=group_schema,
        key=lambda x: len(x["conformations"][0]["atoms"]),
    )
    end = time.time()
    print(f"Sort {split} files execution time {end - start}s.")

    start = time.time()
    sort_collection(
        input_dir=prefix_filename(f"sorted_{split}_files"),
        output_dir=prefix_filename(f"{split}"),
        chunk_size=1000,
        schema=group_schema,
        key=lambda x: len(x["conformations"][0]["atoms"]),
    )
    shutil.rmtree(prefix_filename(f"grouped_{split}"))
    shutil.rmtree(prefix_filename(f"sorted_{split}_files"))

    end = time.time()
    print(f"Sort {split} collection execution time {end - start}s.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process GEOM data.")
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        default="data/mcf/processed_drugs",
        help="Path to the root folder for GEOM data, default: data/mcf/processed_drugs",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="data/mcf/drugs_20",
        help="Path to the root folder for output, default: data/mcf/drugs_20",
    )
    parser.add_argument(
        "-n",
        "--num_confs",
        type=int,
        default=20,
        help="Number of conformations to use per molecule",
    )
    parser.add_argument(
        "-f",
        "--freq",
        type=float,
        default=1.0,
        help="Molecule downsampling (float), default: 1.0 (all molecules)",
    )
    args = parser.parse_args()
    os.makedirs(args.output)
    # Reading pickle files, every pickle file has single SMILES with multiple conformations
    # SMILES in the name of the pickle file

    def prefix_filename(filename, output_root=args.output):
        return os.path.join(output_root, filename)

    (
        train_filenames,
        val_filenames,
        test_filenames,
        test_1000_filenames,
    ) = get_geom_filenames(root=args.data, max_confs=args.num_confs)
    print(f"Number of conformations train {len(train_filenames)}")
    print(f"Number of conformations val {len(val_filenames)}")
    print(f"Number of conformations test {len(test_filenames)}")
    print(f"Number of conformations test_1000 {len(test_1000_filenames)}")

    # Pipeline start, defining schema
    schema = DictionarySchema(
        {
            "dataset": StrSchema,
            "eig_vals": NumpySchema,
            "eig_vecs": NumpySchema,
            "smiles": StrSchema,
            "atoms": ObjectSchema,
            "charges": ObjectSchema,
            "explicitHs": ObjectSchema,
            "graph": NumpySchema,
            "positions": NumpySchema,
        }
    )

    group_schema = DictionarySchema(
        {
            "smiles": StrSchema,
            "conformations": ListSchema(
                DictionarySchema(
                    {
                        "dataset": StrSchema,
                        "eig_vals": NumpySchema,
                        "eig_vecs": NumpySchema,
                        "atoms": ObjectSchema,
                        "charges": ObjectSchema,
                        "explicitHs": ObjectSchema,
                        "graph": NumpySchema,
                        "positions": NumpySchema,
                    }
                )
            ),
        }
    )

    # Convert pickle files into a raw dataset in Nexus format
    # Every item is SMILE and has multiple conformations

    start = time.time()
    build_sort_and_group_dataset(
        train_filenames, "train", schema, group_schema, freq=args.freq
    )
    end = time.time()
    print(f"Build sort and group geom drugs train execution time {end - start}s.")

    start = time.time()
    build_sort_and_group_dataset(val_filenames, "val", schema, group_schema)
    end = time.time()
    print(f"Build sort and group geom drugs val execution time {end - start}s.")

    start = time.time()
    build_sort_and_group_dataset(test_filenames, "test", schema, group_schema)
    end = time.time()
    print(f"Build sort and group geom drugs test execution time {end - start}s.")

    start = time.time()
    build_sort_and_group_dataset(test_1000_filenames, "test_1000", schema, group_schema)
    end = time.time()
    print(f"Build sort and group geom drugs test_1000 execution time {end - start}s.")

    # Convert SMILES item into separate items for each conformation for the Train split
    # See the unrolled_schema below. The ListSchema under key is unrolled.
    # We increase chunk_size as there are 200 conformation for every SMILIES on average
    start = time.time()
    unroll_schema(
        key="conformations",
        new_key="conformation",
        schema=group_schema,
        input_dir=prefix_filename("grouped_train"),
        output_dir=prefix_filename("raw_train_unrolled"),
        chunk_size=60000,
    )
    end = time.time()
    print(f"Unroll train execution time {end - start}s.")

    # Shuffle Train split
    start = time.time()
    shuffle_each_collection_file(
        input_dir=prefix_filename("raw_train_unrolled"),
        output_dir=prefix_filename("raw_train_unrolled_shuffled_files"),
    )
    end = time.time()
    print(f"Shuffle train files execution time {end - start}s.")

    start = time.time()
    shuffle_collection(
        input_dir=prefix_filename("raw_train_unrolled_shuffled_files"),
        output_dir=prefix_filename("train"),
        chunk_size=12000,
    )
    end = time.time()
    shutil.rmtree(prefix_filename("grouped_train"))
    shutil.rmtree(prefix_filename("raw_train_unrolled"))
    shutil.rmtree(prefix_filename("raw_train_unrolled_shuffled_files"))
    print(f"Shuffle train execution time {end - start}s.")

    # Sort Test split based on number of atoms
    sort_grouped_dataset("val", group_schema)
    sort_grouped_dataset("test", group_schema)
    sort_grouped_dataset("test_1000", group_schema)

    # Unrolled schema
    unrolled_schema = DictionarySchema(
        {
            "smiles": StrSchema,
            "conformation": DictionarySchema(
                {
                    "dataset": StrSchema,
                    "eig_vals": NumpySchema,
                    "eig_vecs": NumpySchema,
                    "atoms": ObjectSchema,
                    "charges": ObjectSchema,
                    "explicitHs": ObjectSchema,
                    "graph": NumpySchema,
                    "positions": NumpySchema,
                }
            ),
        }
    )

    # Building index.json files for LitData and place it into input_dir
    # Notice that we use different schemas for train and test splits
    # Schemas are also serialized into index.json
    start = time.time()
    build_litdata_dataset_index(
        input_dir=prefix_filename("train"), schema=unrolled_schema
    )
    end = time.time()
    print(f"Build litdata dataset index train execution time {end - start}s.")

    start = time.time()
    build_litdata_dataset_index(input_dir=prefix_filename("val"), schema=group_schema)
    end = time.time()
    print(f"Build litdata dataset index val execution time {end - start}s.")

    start = time.time()
    build_litdata_dataset_index(input_dir=prefix_filename("test"), schema=group_schema)
    end = time.time()
    print(f"Build litdata dataset index test execution time {end - start}s.")

    start = time.time()
    build_litdata_dataset_index(
        input_dir=prefix_filename("test_1000"), schema=group_schema
    )
    end = time.time()
    print(f"Build litdata dataset index test_1000 execution time {end - start}s.")
