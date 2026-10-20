import os
from functools import partial
from pathlib import Path
from typing import Any

import lightning as pl
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint


def serialize_dataloader_states(state):
    """Serialize and collect dataloader states on rank 0."""
    distributed = ("NODE_RANK" in os.environ) and ("LOCAL_RANK" in os.environ)
    if distributed and (
        not torch.distributed.is_available() or not torch.distributed.is_initialized()
    ):
        raise RuntimeError(
            "Torch distributed is not available to collect dataloader states"
        )

    if not distributed or torch.distributed.get_world_size() == 1:
        state = {"world_size": 1, "state": [state]}
        return state

    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    states = [None] * world_size if rank == 0 else None
    torch.distributed.gather_object(state, states)

    if rank != 0:
        return None

    return_state = {"world_size": world_size, "state": states}
    return return_state


def deserialize_dataloader_states(state):
    """Deserialization of dataloader state, see serialize_dataloader_states."""
    distributed = ("NODE_RANK" in os.environ) and ("LOCAL_RANK" in os.environ)
    checkpoint_world_size = state["world_size"]
    state = state["state"]
    if checkpoint_world_size == 1 and not distributed:
        return state[0]
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        raise RuntimeError(
            f"Dataloader checkpoints require the same number of GPUs: {checkpoint_world_size}."
        )

    world_size = torch.distributed.get_world_size()
    if world_size != checkpoint_world_size:
        raise RuntimeError(
            f"Dataloader checkpoints require the same number of GPUs. You have {checkpoint_world_size} vs {world_size}"
        )

    rank = torch.distributed.get_rank()
    return state[rank]


class SaveLoadStateCallback(pl.Callback):
    dataloader_state_entry_name = "nexus_dataloader_state"
    lm_state_entry_name = "lm_state"

    def __init__(self, datamodule, load_data_state) -> None:
        super().__init__()
        self.datamodule = datamodule
        self.load_data_state = load_data_state

    def on_save_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        checkpoint: dict[str, Any],
    ) -> None:
        if hasattr(pl_module, "get_state_dict"):
            module_state_dict = pl_module.get_state_dict()
            if module_state_dict is not None:
                checkpoint[
                    SaveLoadStateCallback.lm_state_entry_name
                ] = module_state_dict

        dataloader_state = self.datamodule.get_state_dict()
        dataloader_state = serialize_dataloader_states(dataloader_state)
        if dataloader_state:
            checkpoint[
                SaveLoadStateCallback.dataloader_state_entry_name
            ] = dataloader_state

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        if (
            SaveLoadStateCallback.dataloader_state_entry_name in checkpoint
            and self.load_data_state
        ):
            dataloader_state = deserialize_dataloader_states(
                checkpoint[SaveLoadStateCallback.dataloader_state_entry_name]
            )
            self.datamodule.load_state_dict(dataloader_state)
        if SaveLoadStateCallback.lm_state_entry_name in checkpoint and hasattr(
            pl_module, "load_user_states"
        ):
            pl_module.load_user_states(
                checkpoint[SaveLoadStateCallback.lm_state_entry_name]
            )


def checkpointing_callbacks(
    root_dir,
    epoch_dir=None,
    epoch_filename="epoch_{epoch}",
    monitor=None,
    datamodule=None,
    n_steps=None,
    save_datamodule_state=False,
    save_all_steps=True,
    load_datamodule_state=True,
):
    """
    Creates the checkpointing callbacks.

    If `n_steps` is set, a checkpoint is written to `root_dir` every `n_steps`
    training steps and `last.ckpt` is updated, to allow re-running training from
    the last model created. With `save_all_steps`, every one of these
    checkpoints is kept as "epoch={epoch}-step={step}.ckpt"; otherwise only
    `last.ckpt` is kept. If `epoch_dir` is set and `monitor` is set, the two
    checkpoints with the lowest value of `monitor` are kept in `epoch_dir`, with
    the value in the filename. If `epoch_dir` is set and `monitor` is not, a
    checkpoint named by `epoch_filename` is kept for every epoch. If
    `save_datamodule_state` is set, the datamodule state is stored in the
    checkpoints and, with `load_datamodule_state`, restored from them.
    """
    filename = "epoch={epoch}-step={step}"
    callbacks = []
    if n_steps:
        save_top_k = -1 if save_all_steps else 0
        callbacks.append(
            ModelCheckpoint(
                dirpath=Path(root_dir),
                save_last=True,
                save_top_k=save_top_k,
                filename=filename,
                auto_insert_metric_name=False,
                every_n_train_steps=n_steps,
            )
        )

    if epoch_dir:
        if monitor:
            filename = "epoch={epoch}-step={step}-" + monitor + "={" + monitor + ":.2f}"
            callbacks.append(
                ModelCheckpoint(
                    dirpath=Path(epoch_dir),
                    save_top_k=2,
                    monitor=monitor,
                    mode="min",
                    filename=filename,
                    auto_insert_metric_name=False,
                )
            )
        else:
            callbacks.append(
                ModelCheckpoint(
                    dirpath=Path(epoch_dir),
                    save_top_k=-1,
                    filename=epoch_filename,
                    auto_insert_metric_name=False,
                )
            )

    if save_datamodule_state:
        callbacks.append(SaveLoadStateCallback(datamodule, load_datamodule_state))

    return callbacks


class Task:
    primary_checkpoint = "primary_checkpoint"
    last_checkpoint = "last.ckpt"

    def __init__(
        self,
        trainer,
        datamodule,
        module,
        restart=False,
        checkpoint=None,
    ):
        if checkpoint:
            primary_checkpoint = checkpoint.pop(Task.primary_checkpoint, None)
            datamodule.load_state = checkpoint.get("load_datamodule_state", True)
            callbacks = checkpointing_callbacks(datamodule=datamodule, **checkpoint)
        else:
            callbacks = None

        trainer["limit_val_batches"] = 1.0 if datamodule.val_dataloader_config else 0
        self._trainer = Trainer(**trainer, callbacks=callbacks)

        self.restart = False
        if restart and checkpoint:
            step_dir = checkpoint["root_dir"]
            checkpoint = Path(step_dir) / Task.last_checkpoint
            if checkpoint.exists():
                checkpoint_content = torch.load(
                    checkpoint,
                    map_location=None if torch.cuda.is_available() else "cpu",
                    weights_only=False,
                )
                self.checkpoint = checkpoint
                self.restart = True
            elif primary_checkpoint:
                checkpoint = Path(step_dir) / primary_checkpoint
                checkpoint_content = torch.load(
                    checkpoint,
                    map_location=None if torch.cuda.is_available() else "cpu",
                    weights_only=False,
                )
                module.load_state_dict(checkpoint_content["state_dict"])

    @property
    def fit(self):
        if self.restart:
            return partial(
                self._trainer.fit,
                ckpt_path=self.checkpoint,
                weights_only=False,
            )
        else:
            return self._trainer.fit
