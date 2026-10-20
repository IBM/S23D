#!/usr/bin/env python

import argparse
import os

import numpy as np

from smiles23d.nexusdb.shuffle import (
    DictionarySchema,
    ListSchema,
    NumpySchema,
    ObjectSchema,
    StrSchema,
)
from smiles23d.nexusdb.shuffle.filters import filter_dataset_to_index

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build eval indices for GEOM data.")
    parser.add_argument(
        "-r",
        "--root",
        type=str,
        default="data/mcf",
        help="Path to the root folder for GEOM data, default: data/mcf",
    )
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        default="drugs_20",
        help="Name of the data folder to process, default: drugs_20",
    )

    args = parser.parse_args()
    root_folder = os.path.abspath(args.root)

    freq: int | None = None  # 100
    datasets = [args.data]
    splits = ["test", "test_1000", "val"]

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

    indices = [
        os.path.join(root_folder, dataset, "index", "eval_index")
        for dataset in datasets
    ]
    if freq:
        indices = [index + f"_{freq}" for index in indices]
    [os.makedirs(index) for index in indices]

    def func(dataset):
        def get_dataset(x):
            n = len(x["conformations"])
            if 0 <= n <= 10000:
                return n * 2
            return 0

        def get_dataset_freq(x):
            n = len(x["conformations"])
            if 0 <= n <= 10000:
                if np.random.uniform() < 1.0 / freq:
                    return n * 2
            return 0

        if freq:
            return get_dataset_freq
        else:
            return get_dataset

    for dataset, output_folder in zip(datasets, indices):
        for split in splits:
            filter_dataset_to_index(
                input_dir=os.path.join(root_folder, dataset, split),
                output_dir=os.path.join(output_folder, split),
                schema=group_schema,
                filter=func(dataset),
                chunk_size=1000,
            )
