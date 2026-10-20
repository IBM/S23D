#!/usr/bin/env python

"""See README.md for additional description."""

import argparse
import os

import numcodecs

from smiles23d.nexusdb.datasets.nexus_zarr import build_zarr
from smiles23d.nexusdb.shuffle import (
    DictionarySchema,
    ListSchema,
    NumpySchema,
    ObjectSchema,
    StrSchema,
)
from smiles23d.nexusdb.shuffle.filters import filter_dataset_to_index

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Zarr for GEOM data.")
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
        help="Name of the data folder to process into Zarr format, default: drugs_20",
    )

    args = parser.parse_args()
    root_folder = os.path.abspath(args.root)
    datasets = [args.data]
    splits = ["train", "test", "test_1000", "val"]
    split_chunk_sizes = [12000, 1000, 1000, 1000]

    index_folders = [
        os.path.join(root_folder, dataset, "index", "zarr_index")
        for dataset in datasets
    ]
    [os.makedirs(index_folder) for index_folder in index_folders]

    output_dataset_folders = [
        os.path.join(root_folder, dataset, "zarr") for dataset in datasets
    ]
    [os.makedirs(data_folder) for data_folder in output_dataset_folders]

    for dataset, output_dataset in zip(datasets, output_dataset_folders):
        # Conversion of NexusData into Zarr file
        compressor = numcodecs.Blosc(
            cname="zstd", clevel=3, shuffle=numcodecs.Blosc.BITSHUFFLE
        )

        for split, chunk_size in zip(splits, split_chunk_sizes):
            build_zarr(
                input_dir=os.path.join(root_folder, dataset, split),
                output_dir=os.path.join(output_dataset, split),
                chunk_size=chunk_size,
                compressor=compressor,
            )

    # Index creation

    # Pipeline start, defining schema
    unrolled_schema = DictionarySchema(
        {
            "smiles": StrSchema,
            "conformation": DictionarySchema(
                {
                    "dataset": StrSchema,
                    # "mol": ObjectSchema,
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

    group_schema = DictionarySchema(
        {
            "smiles": StrSchema,
            "conformations": ListSchema(
                DictionarySchema(
                    {
                        "dataset": StrSchema,
                        # "mol": ObjectSchema,
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

    def func(dataset):
        def get_dataset(x):
            datasets = {i["dataset"] for i in x["conformations"]}
            return dataset in datasets

        return get_dataset

    for dataset, index_folder in zip(datasets, index_folders):
        filter_dataset_to_index(
            input_dir=os.path.join(root_folder, dataset, "train"),
            output_dir=os.path.join(index_folder, "train"),
            schema=unrolled_schema,
            # filter=lambda x: x["conformation"]["dataset"] == dataset,
            filter=lambda x: True,
            chunk_size=12000,
        )
        filter_dataset_to_index(
            input_dir=os.path.join(root_folder, dataset, "val"),
            output_dir=os.path.join(index_folder, "val"),
            schema=group_schema,
            # filter=func(dataset),
            filter=lambda x: True,
            chunk_size=1000,
        )
        filter_dataset_to_index(
            input_dir=os.path.join(root_folder, dataset, "test"),
            output_dir=os.path.join(index_folder, "test"),
            schema=group_schema,
            # filter=func(dataset),
            filter=lambda x: True,
            chunk_size=1000,
        )
        filter_dataset_to_index(
            input_dir=os.path.join(root_folder, dataset, "test_1000"),
            output_dir=os.path.join(index_folder, "test_1000"),
            schema=group_schema,
            # filter=func(dataset),
            filter=lambda x: True,
            chunk_size=1000,
        )
