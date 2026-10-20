import os
from functools import partial

import lightning as L
import torch
from lightning.pytorch.utilities import rank_zero_warn
from torch.utils.data import DataLoader

from smiles23d.nexusdb.datasets.nexus import NexusDataloader, NexusDataset
from smiles23d.nexusdb.datasets.nexus_zarr import NexusZarrDataloader, NexusZarrDataset
from smiles23d.tokenizers.piletok import PileTokenizer

from .collator import GEOMCollator


class GEOMDatamodule(L.LightningDataModule):
    """
    PyTorch Lightning DataModule for GEOM data, using a dataset in the
    S23D format.

    Args:
    ----
        input_dir (str): The directory containing the input data.
        shuffle (bool): Whether to shuffle the data.
        batch_size (int, optional): The batch size for training.
            Defaults to 2.
        num_workers (int, optional): The number of workers for data
            loading. The test and predict dataloaders always use 0.
            Defaults to 2.
        index_dir (str, optional): The directory containing the index
            data. Defaults to None.
        start_idx (int, optional): The starting index for data loading.
            Defaults to None.
        end_idx (int, optional): The ending index for data loading.
            Defaults to None.
        limit_samples (int, optional): The maximum number of samples to
            load. Defaults to None.
        samples_per_molecule (int, optional): The number of samples per
            molecule. Defaults to 1.
        mask_prob (float, optional): The probability of masking atoms.
            Defaults to None.
        predict_split (str, optional): The split to use for prediction.
            Options: "test", "val", "test_1000". Defaults to "test".
        removeHs (bool, optional): Whether to remove hydrogen atoms
            when processing data items. Defaults to False.
        val_dataloader_config (dict, optional): Extra configuration for
            the validation dataloader when the main config options
            apply to the train_dataloader. The val_dataloader will
            usually have a different config. Defaults to None.
        use_so3 (bool, optional): Whether to use SO(3) rotations.
            Defaults to False.

    Methods:
    -------
        setup(stage: str): Sets up the data module for training with
            stage = "fit" (the only option at the moment). Creates a
            train_dataloader with the class options. Uses
            val_dataloader_config, if provided, to set up a validation
            dataloader with different config options.
        set_tokenizer(tokenizer: PileTokenizer): Sets the tokenizer for
            the data module.
        _get_dataloader(split: str, inference: bool = False) ->
                                NexusDataloader | NexusZarrDataloader:
            Get the dataloader for the specified split. If 'index_dir'
            is provided, use the NexusZarrDataloader. If
            val_dataloader_config is provided and split is 'val', use
            those config options. If inference is True, use
            num_workers=0.
        get_state_dict() -> dict: Returns the state dictionary of the
            train dataloader.
        load_state_dict(state_dict: dict): Loads the state
            dictionary into the train dataloader.
        train_dataloader() -> DataLoader: Returns the train dataloader.
        val_dataloader() -> DataLoader: Returns the validation
            dataloader.
        test_dataloader() -> DataLoader: Same as predict_dataloader.
        predict_dataloader() -> DataLoader: Creates predict dataloader.

    """

    def __init__(
        self,
        input_dir: str,
        shuffle: bool,
        batch_size: int = 2,
        num_workers: int = 2,
        index_dir: str | None = None,
        start_idx: int | None = None,
        end_idx: int | None = None,
        limit_samples: int | None = None,
        samples_per_molecule: int = 1,
        mask_prob: float | None = None,
        predict_split: str = "test",
        removeHs: bool = False,
        val_dataloader_config: dict | None = None,
        use_so3: bool = False,
        load_state: bool = True,
    ):
        super().__init__()
        self.input_dir = input_dir
        self.index_dir = index_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.samples_per_molecule = samples_per_molecule

        self.collator = partial(
            GEOMCollator,
            num_workers=num_workers,
            samples_per_molecule=samples_per_molecule,
            removeHs=removeHs,
            use_so3=use_so3,
        )

        if not start_idx and limit_samples:
            self.start_idx = 0
        self.start_idx = start_idx
        self.end_idx = end_idx

        if limit_samples:
            self.end_idx = self.start_idx + limit_samples

        self.mask_prob = mask_prob
        self.predict_split = predict_split
        self.val_dataloader_config = val_dataloader_config
        self.load_state = load_state

    def setup(self, stage: str):
        if stage == "fit":
            self._train_dataloader = self._get_dataloader(split="train")
            if self.val_dataloader_config:
                self._val_dataloader = self._get_dataloader(split="val")
            else:
                empty_dataset = torch.utils.data.TensorDataset(torch.Tensor([]))
                self._val_dataloader = DataLoader(empty_dataset, batch_size=1)

    def set_tokenizer(self, tokenizer: PileTokenizer):
        self.tokenizer = tokenizer

    def _get_dataloader(
        self, split: str, inference: bool = False
    ) -> NexusDataloader | NexusZarrDataloader:
        if self.val_dataloader_config and split == "val":
            index_dir = self.val_dataloader_config.get("index_dir", self.index_dir)
            index_dir = os.path.join(index_dir, split) if index_dir else index_dir
            input_dir = self.val_dataloader_config.get("input_dir", self.input_dir)
            input_dir = os.path.join(input_dir, f"{split}")

            dataset_class = (
                partial(NexusZarrDataset, index_dir=index_dir)
                if index_dir
                else NexusDataset
            )
            dataloader_class = NexusZarrDataloader if index_dir else NexusDataloader
            batch_size = self.val_dataloader_config.get("batch_size", self.batch_size)
            num_workers = self.val_dataloader_config.get(
                "num_workers", self.num_workers
            )
            start_idx = self.val_dataloader_config.get("start_idx", self.start_idx)
            end_idx = self.val_dataloader_config.get("end_idx", self.end_idx)
        else:
            index_dir = (
                os.path.join(self.index_dir, split)
                if self.index_dir
                else self.index_dir
            )
            input_dir = os.path.join(self.input_dir, f"{split}")

            dataset_class = (
                partial(NexusZarrDataset, index_dir=index_dir)
                if self.index_dir
                else NexusDataset
            )
            dataloader_class = (
                NexusZarrDataloader if self.index_dir else NexusDataloader
            )
            batch_size = self.batch_size
            num_workers = self.num_workers
            start_idx = self.start_idx
            end_idx = self.end_idx

        if inference and num_workers > 0:
            rank_zero_warn(
                f"num_workers={num_workers} drops batches in test and predict "
                "dataloaders; using num_workers=0."
            )
            num_workers = 0

        # Setting up collator
        collator = self.collator(self.tokenizer)
        if num_workers > 0:
            worker_init_fn = collator.setup_rng
        else:
            worker_init_fn = None
            collator.setup_rng(0)

        if split == "train":
            collator.setup_augmentation(mask_prob=self.mask_prob)
            dataset = dataset_class(input_dir, shuffle=self.shuffle)
        else:
            collator.setup_augmentation(mask_prob=0.0)
            collator = collator.predict_collator
            dataset = dataset_class(
                input_dir,
                shuffle=False,
                start_idx=start_idx,
                end_idx=end_idx,
            )

        multiprocessing_context = "fork" if num_workers > 0 else None

        dataloader = dataloader_class(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collator,
            worker_init_fn=worker_init_fn,
            multiprocessing_context=multiprocessing_context,
            load_state=self.load_state,
        )
        return dataloader

    def get_state_dict(self):
        return self._train_dataloader.state_dict()

    def load_state_dict(self, state_dict) -> None:
        self._train_dataloader.load_state_dict(state_dict)

    def train_dataloader(self):
        return self._train_dataloader

    def val_dataloader(self):
        return self._val_dataloader

    def test_dataloader(self):
        return self._get_dataloader(split=self.predict_split, inference=True)

    def predict_dataloader(self):
        return self._get_dataloader(split=self.predict_split, inference=True)
