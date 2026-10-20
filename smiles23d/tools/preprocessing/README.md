# GEOM-DRUGS preprocessing

This runbook builds the `drugs_20` dataset used by the training and evaluation
configs in `configs/`. The input is the GEOM-DRUGS data as preprocessed by MCF
(https://github.com/apple/ml-mcf). The scripts are in `wang2024/`:

| Script | Reads | Writes |
|---|---|---|
| `build_drugs.py` | MCF pickles | `drugs_20/{train,val,test,test_1000}/` |
| `build_zarr.py` | `drugs_20/<split>/` | `drugs_20/zarr/`, `drugs_20/index/zarr_index/` |
| `build_eval_index.py` | `drugs_20/{val,test,test_1000}/` | `drugs_20/index/eval_index/` |

The part of the finished `drugs_20` that the release configs read is also on
Hugging Face at `ibm-research/S23D.GEOM.processed`: the data the released
models were trained and evaluated on (see the
[main README](../../../README.md#data)). The full `test` split and the other
`zarr/` and `index/` folders are not; build them as described here. Rebuilding from the MCF archives reproduces the content but not
the record order. The check below expects a complete build; for a Hugging Face
download, use the dataset's `checksums.sha256` instead.

All commands below are run from the repository root. The downloads and the
build write into the top-level `data/` directory, which git ignores.

## Requirements

- The `smiles23d` package, installed as described in the main README. The
  scripts import `smiles23d` and do not run from a plain checkout.
- `torch`, which is installed with the package. The MCF pickles contain torch
  tensors.
- 10 or more CPU cores recommended. The pickle conversion in `build_drugs.py`
  starts 10 worker processes; it also runs on fewer cores, but more slowly.
- Disk space: 69 GB for the two archives, room for the extracted archive (about
  7.3 million small files, so its size on disk depends on the filesystem's
  block size), about 200 GB at peak for intermediate files during
  `build_drugs.py`, and about 113 GB for the finished dataset (62 GB of `.bin`
  files and 51 GB of Zarr).

## Download

Download the two GEOM-DRUGS archives from MCF (68 GB and 1.2 GB) and extract
them. The test_1000 archive is extracted into `processed_drugs/`.

```sh
mkdir -p data/mcf
wget https://docs-assets.developer.apple.com/ml-research/datasets/mcf/drugs_new/processed_drugs.tar.gz -P data/mcf/
wget https://docs-assets.developer.apple.com/ml-research/datasets/mcf/drugs_new/processed_drugs_test_1000.tar.gz -P data/mcf/
tar -xzf data/mcf/processed_drugs.tar.gz -C data/mcf/
tar -xzf data/mcf/processed_drugs_test_1000.tar.gz -C data/mcf/processed_drugs/
```

`build_drugs.py` reads the `*.pkl` files in `data/mcf/processed_drugs/<split>/`,
so this must list `test`, `test_1000`, `train` and `val`:

```sh
ls data/mcf/processed_drugs
```

If a folder is missing or nested one level deeper, move it into place. The
archives can be deleted after extraction.

## Build

```sh
python smiles23d/tools/preprocessing/wang2024/build_drugs.py \
    --data data/mcf/processed_drugs --output data/mcf/drugs_20 --num_confs 20
python smiles23d/tools/preprocessing/wang2024/build_zarr.py --root data/mcf --data drugs_20
python smiles23d/tools/preprocessing/wang2024/build_eval_index.py --root data/mcf --data drugs_20
```

`build_drugs.py` creates `--output` and stops if it already exists.
`build_zarr.py` and `build_eval_index.py` take the parent folder as `--root`
and the dataset folder name as `--data`. `build_drugs.py` also has `--freq`
(default 1.0), which randomly drops training molecules. The expected values
below assume `--num_confs 20` and the default `--freq`.

What the build keeps:

- Conformers are grouped into molecules by the `smi` key of the MCF pickles.
- train and val: only conformer files whose index (the number after the last
  `_` in the file name) is below `--num_confs`, so at most 20 conformers per
  molecule. test and test_1000 keep all conformers.
- All splits: conformers whose bond graph is disconnected, and conformers with
  one atom or fewer, are dropped. For test_1000 this reduces 106,775 raw
  conformers to 104,152.

Resulting layout:

```
drugs_20/
  train/        data*.bin, index.json   one item per conformer, shuffled
  val/          data*.bin, index.json   one item per molecule, ordered by atom count
  test/         data*.bin, index.json   same as val
  test_1000/    data*.bin, index.json   same as val
  zarr/         train/, val/, test/, test_1000/
  index/
    zarr_index/ train/, val/, test/, test_1000/   one entry per item; not used by the configs
    eval_index/ val/, test/, test_1000/           chunk_index*.bin, index.json
```

Each `eval_index` entry is the position of a molecule in its split. A molecule
with n kept conformers appears 2n times, so the evaluation generates two
conformers per reference conformer (test_1000: 2 x 104,152 = 208,304 entries).

## Check

Run this check after every build. `build_drugs.py` does not check the exit
status of its worker processes. If a worker dies, for example because it runs
out of memory, the pickles queued for it are skipped, the later steps still run
without an error, and the script may then hang instead of exiting. The counts
below are the only reliable way to detect lost data.

Save the following as `check_drugs20.py`. It uses only the Python standard
library. For each split it reads `index.json` and compares the number of
chunks, the number of items, and for the data splits the total size of the
`.bin` files.

```python
"""Self-check for a finished drugs_20 build (standard library only)."""
import json
import os
import sys

root = sys.argv[1]

# folder: (chunks, items, total bytes of the .bin files, or None)
expected = {
    "train": (342, 4_103_418, 46_786_319_816),
    "val": (31, 30_414, 5_799_544_884),
    "test": (31, 30_403, 8_217_152_771),
    "test_1000": (1, 1_000, 1_442_951_338),
    "index/eval_index/test_1000": (209, 208_304, None),
    "index/eval_index/val": (1_025, 1_024_072, None),
    "index/eval_index/test": (1_421, 1_420_512, None),
}

failed = False
for rel, want in expected.items():
    folder = os.path.join(root, rel)
    try:
        with open(os.path.join(folder, "index.json")) as f:
            chunks = json.load(f)["chunks"]
        n_bytes = None
        if want[2] is not None:
            n_bytes = sum(
                os.path.getsize(os.path.join(folder, c["filename"])) for c in chunks
            )
        got = (len(chunks), sum(c["chunk_size"] for c in chunks), n_bytes)
    except (OSError, ValueError, KeyError) as e:
        got = repr(e)
    ok = got == want
    failed = failed or not ok
    print("OK    " if ok else "FAILED", rel, "got", got, "expected", want)

for split in ("train", "val", "test", "test_1000"):
    if not os.path.isdir(os.path.join(root, "zarr", split)):
        failed = True
        print("FAILED", f"zarr/{split}", "missing")

sys.exit(1 if failed else 0)
```

```sh
python check_drugs20.py data/mcf/drugs_20
```

Every line must print `OK` and the script must exit with status 0. If a data
split fails, delete `drugs_20/` and rebuild everything. If only an
`eval_index` line fails, rebuild the eval index; if only a `zarr` line fails,
rebuild the Zarr arrays (see Rebuilding). The `zarr` lines only confirm that
each split was created; a `build_zarr.py` run that ended without an error is
complete.

## Use with the configs

The configs read the dataset location from the `SMILES23D_DATA` environment
variable, which must point to the `drugs_20` folder:

```sh
export SMILES23D_DATA=$PWD/data/mcf/drugs_20
```

| Config | `input_dir` | `index_dir` | Other settings |
|---|---|---|---|
| `s23d_*_stage1`, `s23d_*_stage2` (training) | `${SMILES23D_DATA}` | not set | `shuffle: False` |
| `s23d_*_eval2k` (evaluation) | `${SMILES23D_DATA}/zarr` | `${SMILES23D_DATA}/index/eval_index` | `predict_split: test_1000`, `start_idx: 0`, `end_idx: 208304` |

A complete evaluation on test_1000 prints `n rmsd calcs: 77763496` to standard
output in the RMSD step. A smaller number means that generated samples are missing or could not
be converted to molecules.

## Rebuilding

Each script refuses to write into folders that already exist:

| Script | Fails if this exists | Remove before rerunning |
|---|---|---|
| `build_drugs.py` | `drugs_20/` | `drugs_20/` (this also removes `zarr/` and `index/`) |
| `build_zarr.py` | `drugs_20/zarr/` or `drugs_20/index/zarr_index/` | both |
| `build_eval_index.py` | `drugs_20/index/eval_index/` | `drugs_20/index/eval_index/` |

An interrupted `build_drugs.py` run leaves intermediate folders in `--output`;
delete the whole folder and start again. The eval index stores positions in
the splits, so `zarr/` and `index/` are only valid for the `build_drugs.py` run
they were built from; never combine outputs of different runs.

```sh
rm -rf data/mcf/drugs_20/zarr data/mcf/drugs_20/index/zarr_index        # before build_zarr.py
rm -rf data/mcf/drugs_20/index/eval_index                               # before build_eval_index.py
```

A rebuild produces the same molecules and conformers, but in a different order:
the training split is shuffled without a fixed seed, and the worker processes
finish in no fixed order. The training configs use `shuffle: False`, so the
training order follows the stored order and changes with a rebuild. The counts
and sizes in the check do not depend on the order.
