# S23D: SMILES to 3D

`smiles23d` is the code for S23D, described in the preprint
[A standard transformer and attention with linear biases for molecular conformer generation](https://arxiv.org/abs/2506.19834)
by Viatcheslav Gurev and Timothy Rumbell.

S23D generates 3D molecular conformers with a diffusion model whose score
network is a standard, non-equivariant transformer. A graph encoder turns atom
tokens and the 2D molecular graph into atom representations, and a structural
network combines them with the 3D coordinates to predict the score. Graph
structure enters attention through a relative positional encoding: a negative
bias that grows linearly with the shortest-path distance between atoms, with a
different slope per head (an adaptation of ALiBi). Training has two stages:
pretraining on molecules without hydrogens, then fine-tuning on complete
molecules.

| Model | Parameters | Configs | Hugging Face |
|---|---|---|---|
| S23D-S | 8.55 M | `configs/s23d_s_*.yaml` | `ibm-research/S23D.8m.v1` |
| S23D-B | 24.8 M | `configs/s23d_b_*.yaml` | `ibm-research/S23D.25m.v1` |

## Installation

Create a conda environment with Python 3.11 and install S23D into it:

```bash
conda create -n s23d python=3.11 -y
conda activate s23d
git clone https://github.com/IBM/S23D.git
cd S23D
pip install .
```

This installs the `smiles23d` package with its dependencies, including
PyTorch, and the `s23dtrain` and `s23dtest` commands. On Linux the PyTorch
wheels include the CUDA libraries, so running on a GPU needs an NVIDIA driver
recent enough for the installed PyTorch release. Python 3.11 or later is
required.

On a GPU machine, this prints `True` when the driver is recent enough:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

To run the tests:

```bash
pip install ".[test]"
pytest
```

## Quickstart

Generate conformers with a pretrained model from Hugging Face:

```python
from rdkit import Chem
from smiles23d.huggingface import S23DDiffusionHF

model = S23DDiffusionHF.from_pretrained("ibm-research/S23D.8m.v1", map_location="cuda")  # or "cpu"
model.eval()
model.set_predict_parameters(kind="VPSDE", n_steps=300)  # 300 steps, as in the paper

conformers = model.predict("CC(=O)Nc1ccc(O)cc1", samples_per_molecule=10)

# conformers[0] is a list of RDKit molecules, one conformer each.
with Chem.SDWriter("conformers.sdf") as writer:
    for m in conformers[0]:
        writer.write(m)
```

- `predict` takes a SMILES string, an RDKit `Mol`, or a list of either. The
  models were trained with hydrogens, so `predict` adds the hydrogens the input
  leaves implicit and generates coordinates for every atom, in the atom order
  `Chem.AddHs` gives.
- Each input must be one connected molecule, because the models were trained
  on connected molecules only; inputs with several fragments, such as salts,
  are not rejected. The supported elements are B, Bi, Br, C, Cl, F, H, I, N, O,
  P, S and Si; atoms of any other element are encoded as the mask token,
  without a warning.
- An input without a conformer gets one from RDKit ETKDG first; when `predict`
  adds no hydrogens, as after `Chem.AddHs`, that conformer is added to the `Mol`
  you pass in. An invalid SMILES, or a molecule RDKit cannot embed, raises
  `ValueError`.
- With the default `output_format="mol"`, the result has one entry per distinct
  input SMILES, in input order; repeated inputs are merged into one entry.
- `batch_size` (default 128) counts input molecules per forward pass, each
  repeated `samples_per_molecule` times; lower it when you ask for many samples.
- Without `set_predict_parameters`, `predict` uses 200 diffusion steps.
  `output_format="s23d"` returns one dict per input, not merged, with the S23D
  `molecule` and its `predictions`, a list of atom coordinates per sample.

## Data

Training and evaluation use GEOM-DRUGS as preprocessed by
[MCF](https://github.com/apple/ml-mcf), converted into this package's format: a
`drugs_20` directory. The prebuilt data is on Hugging Face at
[`ibm-research/S23D.GEOM.processed`](https://huggingface.co/datasets/ibm-research/S23D.GEOM.processed). The
evaluation needs only its benchmark files (212 data files plus
`checksums.sha256`, 0.62 GB). Download them and check them against the
checksums:

```bash
hf download ibm-research/S23D.GEOM.processed --repo-type dataset \
    --include "zarr/test_1000/*" --include "index/eval_index/test_1000/*" \
    --include checksums.sha256 --local-dir /abs/path/drugs_20
(cd /abs/path/drugs_20 && grep -E '  (zarr|index/eval_index)/test_1000/' checksums.sha256 | sha256sum --check --quiet)
```

For training, download everything (589 data files, 54.7 GB) and check it:

```bash
hf download ibm-research/S23D.GEOM.processed --repo-type dataset --local-dir /abs/path/drugs_20
(cd /abs/path/drugs_20 && sha256sum --check --quiet checksums.sha256)
```

`smiles23d` decodes the records with msgpack-numpy, which can unpickle Python
objects, so only read data that passes this check.

All configs read the data location from the `SMILES23D_DATA` environment
variable. Use an absolute path, because each run starts in its own directory:

```bash
export SMILES23D_DATA=/abs/path/drugs_20
```

The released models were trained and evaluated on this data; its dataset card
lists the layout and the checksums. The Hub copy is the subset the release
configs read: the full `test` split and the other `zarr/` and `index/` folders
need a rebuild from the MCF archives, as described in
[smiles23d/tools/preprocessing/README.md](smiles23d/tools/preprocessing/README.md).
A rebuild reproduces the content but not the record order.

## Configs

| Config | Purpose |
|---|---|
| `s23d_s_stage1` | S23D-S stage 1: no hydrogens, `batch_size` 128 per GPU, `max_epochs` 100 |
| `s23d_s_stage2` | S23D-S stage 2: explicit hydrogens, `batch_size` 128 per GPU, `max_epochs` 135 |
| `s23d_b_stage1` | S23D-B stage 1: no hydrogens, `batch_size` 64 per GPU, `max_epochs` 100 |
| `s23d_b_stage2` | S23D-B stage 2: explicit hydrogens, `batch_size` 64 per GPU, `max_epochs` 130 |
| `s23d_s_eval2k` | GEOM-DRUGS evaluation on `test_1000`, results in `outputs/s23d_s_eval2k`; the architecture comes from the checkpoint |
| `s23d_b_eval2k` | Same as `s23d_s_eval2k`, with results in `outputs/s23d_b_eval2k` |

## Evaluation (GEOM-DRUGS)

Evaluate a published model by its Hugging Face repo id:

```bash
export SMILES23D_DATA=/abs/path/drugs_20
REPO=/path/to/S23D
mkdir -p eval && cd eval
s23dtest -cd "$REPO/configs" -cn s23d_s_eval2k test.checkpoint=ibm-research/S23D.8m.v1
```

or a checkpoint from your own training run (see [Training](#training)) by its path:

```bash
s23dtest -cd "$REPO/configs" -cn s23d_s_eval2k \
    test.checkpoint=/path/to/runs/s23d_s/stage2/epoch_checkpoints/s23d_s_epoch_134.ckpt \
    test.save_path=outputs/s23d_s_ep134
```

For S23D-B use `-cn s23d_b_eval2k` with `ibm-research/S23D.25m.v1` or its
final stage-2 checkpoint, `s23d_b_epoch_129.ckpt`.

On a GPU node without internet access, download the model beforehand on a
machine that has it and pass the folder:

```bash
hf download ibm-research/S23D.8m.v1 --local-dir /abs/path/S23D-S
s23dtest -cd "$REPO/configs" -cn s23d_s_eval2k test.checkpoint=/abs/path/S23D-S
```

- **Data.** The evaluation reads only `zarr/test_1000` and
  `index/eval_index/test_1000` from `$SMILES23D_DATA`; the eval-only download in
  [Data](#data) fetches just these two folders and the checksums.
- **Protocol.** For a test molecule with K reference conformers the model
  generates 2K conformers. The index in `$SMILES23D_DATA/index/eval_index`
  lists each molecule 2K times, so `samples_per_molecule` stays 1.
- **Sampling.** The eval configs use VPSDE with 300 steps
  (`test.diffusion_kwargs`); `predict` defaults to 200 steps.
- **Quoting.** Hydra splits overrides at `=`, so a checkpoint path that contains
  `=`, such as a step checkpoint in `checkpoints/`, must be in double quotes
  inside single quotes: `'test.checkpoint="/path/to/epoch=102-step=1650000.ckpt"'`.
- **Use a new `save_path` for each checkpoint.** If `save_path` already holds
  `mols.pkl` or `rmsds.pkl`, `s23dtest` reuses them and reports the metrics of
  the earlier run.
- **Outputs.** `save_path` receives `output_0.jsonl` (generated conformers),
  `mols.pkl` (RDKit molecules) and `rmsds.pkl` (metrics and RMSD matrices).
  Coverage and AMR, for recall and precision, are printed at the end. RMSDs use
  heavy atoms after alignment, with a coverage threshold of 0.75 Å and no
  chirality correction.
- **Check.** A complete evaluation prints `n rmsd calcs: 77763496` to standard
  output and writes 208,304 rows to `output_0.jsonl`. The `rmsd batch size`
  printed on the same line depends on `test.n_cpus`, so it is not part of the
  check.
- **Versions.** The evaluation results were checked with posebusters 0.6.5,
  rdkit 2026.3.6, numpy 2.4.6, torch 2.14, zarr 3.0.10 and litdata 0.2.30.
- **Resources.** One GPU. A full evaluation takes many hours on a single GPU,
  roughly 10 to 16 h for S23D-S and about 16 h for S23D-B depending on the GPU
  and `test.n_cpus`, and peaks at about 57 GB of host memory with the default
  `test.n_cpus` of 32. Molecule building and RMSD
  computation use `test.n_cpus` processes; lower it to match your machine, for
  example `test.n_cpus=8`. On a shared machine, set `OMP_NUM_THREADS` to the
  number of cores allocated to you, since PyTorch otherwise starts a thread per
  visible core.
- **Checkpoints.** `test.checkpoint` takes a Hugging Face repo id, a directory
  written by `save_pretrained`, or a Lightning checkpoint (`.ckpt`) from a
  training run.

### Expected results

These are the GEOM-DRUGS results reported in the preprint, as mean / median;
coverage uses the 0.75 Å threshold. `s23dtest` prints the same eight numbers as
`Recall Cov Mean`, `Recall Cov Median`, `Recall AMR Mean` and so on, with
coverage in percent. Sampling is stochastic, so a rerun differs slightly.

| Model | Recall COV (%) | Recall AMR (Å) | Precision COV (%) | Precision AMR (Å) |
|---|---|---|---|---|
| S23D-S | 80.7 / 89.9 | 0.483 / 0.458 | 57.5 / 57.5 | 0.757 / 0.700 |
| S23D-B | 84.5 / 91.7 | 0.420 / 0.387 | 62.5 / 63.4 | 0.690 / 0.626 |

## Training

The example trains S23D-S; for S23D-B use `s23d_b_stage1`, `s23d_b_stage2` and
`s23d_b_epoch_99.ckpt`.
Each stage runs in its own directory, so `-cd` points at the repository's
`configs/` directory by an absolute path. A batch job must activate the
environment first, or call `s23dtrain`, `s23dtest` and `python` by their full
path in the environment's `bin/` directory.

```bash
export SMILES23D_DATA=/abs/path/drugs_20
REPO=/path/to/S23D
WORK=/path/to/runs/s23d_s

# Stage 1: no hydrogens
mkdir -p "$WORK/stage1" && cd "$WORK/stage1"
s23dtrain -cd "$REPO/configs" -cn s23d_s_stage1 +task.trainer.devices=2

# Initialize the hydrogen embedding and place the result where stage 2 resumes from
mkdir -p "$WORK/stage2/checkpoints"
python "$REPO/checkpoint_editing/edit_hydrogen_embedding.py" \
    -c "$WORK/stage1/epoch_checkpoints/s23d_s_epoch_99.ckpt" \
    -o "$WORK/stage2/checkpoints/last.ckpt"

# Stage 2: explicit hydrogens
cd "$WORK/stage2"
s23dtrain -cd "$REPO/configs" -cn s23d_s_stage2 +task.trainer.devices=2
```

Stage 1 never sees a hydrogen, so the hydrogen embedding is untrained.
`edit_hydrogen_embedding.py` copies the trained mask-token embedding into the
hydrogen embedding and keeps the rest of the checkpoint, including the optimizer
state and the epoch counter. Edit the epoch checkpoint, as shown:
`checkpoints/last.ckpt` is only written every 50,000 steps, so at the end of
stage 1 it is behind epoch 99. The script is in the clone, not in the installed
package, and runs on CPU with the environment's `python`.

- **Epochs.** Epochs are numbered from 0, so 100 epochs end at
  `s23d_s_epoch_99.ckpt`. Stage 2 resumes the full stage-1 checkpoint, so its
  `task.trainer.max_epochs` counts from the start of stage 1: 35 fine-tuning
  epochs for S23D-S and 30 for S23D-B. If you shorten stage 1 to E epochs with
  `task.trainer.max_epochs=E`, edit `s23d_s_epoch_{E-1}.ckpt` and set stage 2's
  `task.trainer.max_epochs` to E plus the fine-tuning epochs.
- **GPUs and batch size.** Set the number of GPUs with
  `+task.trainer.devices=N`; without it Lightning uses every visible GPU.
  `datamodule.batch_size` is per GPU. The released models were trained on 2
  GPUs, a global batch of 256 for S23D-S and 128 for S23D-B; on a different
  number of GPUs, adjust `datamodule.batch_size` to keep the global batch.
- **Outputs.** Written relative to the launch directory: `checkpoints/` (every
  50,000 steps, plus `last.ckpt`), `epoch_checkpoints/` (one per epoch,
  `s23d_s_epoch_N.ckpt` or `s23d_b_epoch_N.ckpt`, set by
  `task.checkpoint.epoch_filename`) and `lightning_logs/`. Hydra writes the
  resolved config and a log to `outputs/<date>/<time>/`. Use one directory per
  run.
- **Restarts.** Running the same command again in the same directory resumes
  from `checkpoints/last.ckpt`. In stage 2 the seed checkpoint occupies
  `last.ckpt`, so Lightning writes its own as `last-v1.ckpt`; restart stage 2
  with:

  ```bash
  export SMILES23D_DATA=/abs/path/drugs_20
  REPO=/path/to/S23D
  WORK=/path/to/runs/s23d_s
  cd "$WORK/stage2"
  if [ -f checkpoints/last-v1.ckpt ]; then mv checkpoints/last-v1.ckpt checkpoints/last.ckpt; fi
  s23dtrain -cd "$REPO/configs" -cn s23d_s_stage2 +task.trainer.devices=2
  ```

  If `last.ckpt` is missing when stage 2 starts, it trains from random weights
  without an error.
- **Validation** is off. To turn it on, set `datamodule.val_dataloader_config`
  to a mapping such as `{num_workers: 0}`; the other data-loader settings
  default to the training values, and the data is read from `<input_dir>/val`.
  Keep `num_workers: 0`, since more workers drop batches. Validation samples
  full conformers (200 steps) and computes RMSD metrics, so it is slow and not
  directly comparable to `s23dtest`.
- **Logging.** ClearML logging is off unless the config has a top-level
  `clearml:` block with `project_name` and `task_name`.
- **Reproducibility.** The augmentation (random orthogonal transforms and atom
  masking) is seeded from the wall clock, so runs differ.

## Converting a checkpoint to Hugging Face format

```python
from smiles23d.huggingface import S23DDiffusionHF
from smiles23d.tokenizers import PileTokenizer

model = S23DDiffusionHF.load_from_checkpoint(
    "epoch_checkpoints/s23d_s_epoch_134.ckpt",
    tokenizer=PileTokenizer(),
    map_location="cpu",
    weights_only=False,
)
model.save_pretrained("s23d_s_hf")  # or model.push_to_hub("<user>/<repo>")
```

The directory holds `model.safetensors`, `config.json` and a model card, and
works with `from_pretrained` and `s23dtest`.

## Example notebook

[`notebooks/inference_test.ipynb`](notebooks/inference_test.ipynb) loads a
pretrained model from Hugging Face, generates conformers for three drug
molecules and shows them in 3D. Jupyter is not installed with `smiles23d`;
install it separately to run the notebook.

## Citation

If you use S23D, please cite:

```bibtex
@misc{gurev2025standard,
  title         = {A standard transformer and attention with linear biases for molecular conformer generation},
  author        = {Gurev, Viatcheslav and Rumbell, Timothy},
  year          = {2025},
  eprint        = {2506.19834},
  archivePrefix = {arXiv},
  primaryClass  = {q-bio.BM},
  url           = {https://arxiv.org/abs/2506.19834}
}
```

## License

Apache License 2.0. See [LICENSE](LICENSE).
