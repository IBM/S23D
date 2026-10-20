import glob
import importlib
import pickle
from pathlib import Path

import jsonlines
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import BasePredictionWriter, Callback

from smiles23d.huggingface import S23DDiffusionHF
from smiles23d.nexusnet.tasks.maintrainer import (
    deserialize_dataloader_states,
    serialize_dataloader_states,
)
from smiles23d.tools import MolData, RMSDWrapper


class S23DPredictJsonlWriter(BasePredictionWriter):
    """
    Write S23D predictions to a JSONL file.

    Args:
    ----
        save_file_prefix (str): The prefix for the save file.
        write_interval (str, optional): The interval at which to write
            predictions. Defaults to "batch".
        remove_output (bool, optional): Whether to remove existing
            output files. Defaults to False.

    Attributes:
    ----------
        save_file_prefix (str): The prefix for the save file.
        write_interval (str): The interval at which to write
            predictions.
        remove_output (bool): Whether to remove existing output files.
        smiles_collection (list): The list of unique SMILES strings
            passed in the predictions so far. This is to group
            conformations by SMILES in the output, and ensure the full
            list of true conformations is only written to the file once
            for each SMILES, to save disk space.

    Methods:
    -------
        write_on_batch_end(trainer, pl_module, predictions,
        batch_indices, batch, batch_idx, dataloader_idx):
            Writes predictions to a JSONL file at the end of each
            prediction batch.
    """

    def __init__(
        self,
        save_file_prefix: str,
        write_interval: str = "batch",
        remove_output: bool = False,
    ):
        super().__init__(write_interval)
        self.save_file_prefix = save_file_prefix
        if not Path(self.save_file_prefix).parent.exists():
            Path(self.save_file_prefix).parent.mkdir(parents=True)
        self.smiles_collection = []
        if remove_output:
            outfiles = glob.glob(f"{save_file_prefix}*")
            for outfile in outfiles:
                Path(outfile).unlink(missing_ok=True)

    def write_on_batch_end(
        self,
        trainer,
        pl_module,
        predictions,
        batch_indices,
        batch,
        batch_idx,
        dataloader_idx,
    ):
        with jsonlines.open(
            f"{self.save_file_prefix}_{trainer.global_rank}.jsonl", mode="a"
        ) as writer:
            for mol in flatten(predictions):
                if mol["molecule"]["smiles"] in self.smiles_collection:
                    mol["molecule"]["conformations"] = [
                        mol["molecule"]["conformations"][0]
                    ]
                else:
                    self.smiles_collection.append(mol["molecule"]["smiles"])
                writer.write(mol)


class SaveLoadTestDataloaderStateCallback(Callback):
    """
    Callback to save and load the state of the test dataloader.

    Args:
    ----
        save_path (str | Path, optional): Directory for the dataloader
            state file. Defaults to the current working directory.

    Attributes:
    ----------
        dataloader_state_entry_name (str): The name of the entry in the
            state dictionary to save the dataloader state.

    Methods:
    -------
        on_predict_batch_end(trainer, pl_module, outputs, batch,
                             batch_idx): Hook to save the state of
            the test dataloader after each predict batch.
        on_predict_start(trainer, pl_module): Hook to load the
            state of the test dataloader at the start of prediction.
        on_predict_end(trainer, pl_module): Hook to delete the saved
            state once prediction has finished.
    """

    dataloader_state_entry_name = "nexus_test_dataloader_state"

    def __init__(self, save_path: str | Path = ".") -> None:
        super().__init__()
        self.save_path = save_path

    def _state_file(self, trainer) -> Path:
        return Path(
            self.save_path,
            f"{self.dataloader_state_entry_name}_{trainer.global_rank}.pt",
        )

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        dataloader_state = trainer.predict_dataloaders.state_dict()
        dataloader_state = serialize_dataloader_states(dataloader_state)
        if dataloader_state:
            torch.save(dataloader_state, self._state_file(trainer))

    def on_predict_start(self, trainer, pl_module):
        print("Trying to resume datamodule state")
        try:
            state = torch.load(self._state_file(trainer))
        except FileNotFoundError:
            print("Saved datamodule state not found, skipping...")
            return
        dataloader_state = deserialize_dataloader_states(state)
        trainer.predict_dataloaders.load_state_dict(dataloader_state)

    def on_predict_end(self, trainer, pl_module):
        self._state_file(trainer).unlink(missing_ok=True)


class S23DPredict:
    """
    S23DPredict class for predicting using S23D model, a saved
    checkpoint, and a datamodule with a val or test set.

    Args:
    ----
        checkpoint (str): Path to a Lightning checkpoint (`.ckpt`), or
            a Hugging Face repo id or `save_pretrained` directory.
        tokenizer (Tokenizer): Tokenizer object for tokenizing input.
        diffusion_kwargs (dict): Keyword arguments for diffusion model.
        save_path (str, Path, optional): Path to save the output.
            Defaults to None, which sets the current working directory.
        save_jsonl_file_prefix (str, optional): Prefix for the JSONL file. Defaults to 'output'.
        save_mol_file (str, Path, optional): Path to save the MolData
            object to a pickle file. Defaults to None.
        save_rmsd_file (str, Path, optional): Path to save the RMSD
            file. Defaults to None.
        trainer (dict, optional): Keyword arguments for trainer.
            Defaults to {}.
        save_dataloader_state (bool, optional): Whether to save the
            dataloader state. Defaults to False.
        remove_output (bool, optional): Whether to remove the output
            files at the start of the prediction. Defaults to False.
        skip_generation (bool, optional): Whether to skip generation in
            `test` and re-score the conformers in an existing
            `<save_jsonl_file_prefix>_0.jsonl` in `save_path`. Needs
            `save_mol_file` or `save_rmsd_file`. Defaults to False.
        n_cpus (int, optional): Number of processes used to build the
            molecules and compute the RMSD metrics. Defaults to 1.
        clearml_logger (optional): ClearML logger to which `test`
            reports the RMSD metrics. Defaults to None.

    Methods:
    -------
        load_checkpoint(module): Load the model from self.checkpoint. A
            `.ckpt` file loads as the class of `module`, or else the
            class named in the checkpoint. A Hugging Face repo id or
            `save_pretrained` directory loads as S23DDiffusionHF with
            its own PileTokenizer, ignoring `module` and `tokenizer`.
        get_trainer(): Create a pytorch Trainer object using any
            callbacks defined by the various save paths, and the
            trainer config dictionary 'trainer'.
        test(datamodule): Run the model to get predictions using the
            predict_step and the datamodule's predict dataloader, unless
            `skip_generation` is set. If `save_mol_file` or
            `save_rmsd_file` is set, then compute and print the RMSD
            metrics and report them to `clearml_logger`, if given.
        validate(datamodule): Run the model's validation_step on the
            validation dataloader, which samples conformers and logs the
            RMSD metrics.
    """

    def __init__(
        self,
        checkpoint,
        tokenizer,
        diffusion_kwargs,
        save_path: str | Path | None = None,
        save_jsonl_file_prefix: str = "output",
        save_mol_file: str | Path | None = None,
        save_rmsd_file: str | Path | None = None,
        trainer: dict = {},
        save_dataloader_state=False,
        remove_output=False,
        skip_generation=False,
        n_cpus=1,
        clearml_logger=None,
    ):
        self.checkpoint = checkpoint
        self.save_path = save_path or Path.cwd().resolve()
        self.save_jsonl_file_prefix = save_jsonl_file_prefix
        self.save_mol_file = save_mol_file
        self.save_rmsd_file = save_rmsd_file
        self.skip_generation = skip_generation
        self.tokenizer = tokenizer
        self.trainer = trainer
        self.save_dataloader_state = save_dataloader_state
        self.diffusion_kwargs = diffusion_kwargs
        self.remove_output = remove_output
        self.n_cpus = n_cpus
        self.clearml_logger = clearml_logger

    def load_checkpoint(self, module=None):
        if not str(self.checkpoint).endswith(".ckpt"):
            self.model = S23DDiffusionHF.from_pretrained(self.checkpoint)
            self.model.set_predict_parameters(**self.diffusion_kwargs)
            return
        if not module:
            cp = torch.load(
                self.checkpoint,
                map_location=None if torch.cuda.is_available() else "cpu",
                weights_only=False,
            )
            module_path, class_name = cp["class_path"].rsplit(".", 1)
            module = importlib.import_module(module_path)
            ModelClass = getattr(module, class_name)
        else:
            ModelClass = module.__class__
        self.model = ModelClass.load_from_checkpoint(
            self.checkpoint, tokenizer=self.tokenizer, weights_only=False
        )
        self.model.set_predict_parameters(**self.diffusion_kwargs)

    def get_trainer(self):
        callbacks = []
        if self.save_jsonl_file_prefix:
            callbacks.append(
                S23DPredictJsonlWriter(
                    save_file_prefix=Path(self.save_path, self.save_jsonl_file_prefix),
                    remove_output=self.remove_output,
                )
            )
        if self.save_dataloader_state:
            callbacks.append(SaveLoadTestDataloaderStateCallback(self.save_path))
        return Trainer(callbacks=callbacks, **self.trainer)

    def test(self, datamodule):
        analysis = bool(self.save_mol_file or self.save_rmsd_file)

        if not self.skip_generation:
            trainer = self.get_trainer()
            predictions = trainer.predict(model=self.model, datamodule=datamodule)
            mols = [p for ps in predictions for p in ps]
        elif analysis:
            # Basic attempt to read from a file, otherwise throw a no-file error
            generated_mol_file_path = Path(
                self.save_path, f"{self.save_jsonl_file_prefix}_0.jsonl"
            )
            with jsonlines.open(generated_mol_file_path, mode="r") as reader:
                mols = list(reader)

        if analysis:
            mol_data = None
            mol_file_path = (
                Path(self.save_path, self.save_mol_file) if self.save_mol_file else None
            )

            if mol_file_path and mol_file_path.exists():
                try:
                    with open(mol_file_path, "rb") as f:
                        mol_data = pickle.load(f)
                except Exception:
                    pass  # Fall back to generating mol_data if loading fails
            if mol_data is None:
                mol_data = MolData(n_cpus=self.n_cpus)
                mol_data.load_molecules(mols)
                if mol_file_path:
                    with open(mol_file_path, "wb") as f:
                        pickle.dump(mol_data, f)

            rmsd_metrics = None
            rmsd_file_path = (
                Path(self.save_path, self.save_rmsd_file)
                if self.save_rmsd_file
                else None
            )

            if rmsd_file_path and rmsd_file_path.exists():
                try:
                    with open(rmsd_file_path, "rb") as f:
                        rmsd_metrics = pickle.load(f)
                except Exception:
                    pass  # Fall back to generating rmsd_metrics if loading fails

            if rmsd_metrics is None:
                rmsd_wrapper = RMSDWrapper(
                    mol_data=mol_data,
                    n_cpus=self.n_cpus,
                    return_rmsds=True,
                )
                rmsd_metrics = rmsd_wrapper()
                if rmsd_file_path:
                    with open(rmsd_file_path, "wb") as f:
                        pickle.dump(rmsd_metrics, f)

            rmsd_metrics_to_log = {
                k: v for k, v in rmsd_metrics.items() if k != "all_mol_rmsds"
            }

            print(f"Recall Cov Mean: {rmsd_metrics['CovR_mean']*100}")
            print(f"Recall Cov Median: {rmsd_metrics['CovR_med']*100}")
            print(f"Recall AMR Mean: {rmsd_metrics['MatR_mean']}")
            print(f"Recall AMR Median: {rmsd_metrics['MatR_med']}")
            print(f"Precision Cov Mean: {rmsd_metrics['CovP_mean']*100}")
            print(f"Precision Cov Median: {rmsd_metrics['CovP_med']*100}")
            print(f"Precision AMR Mean: {rmsd_metrics['MatP_mean']}")
            print(f"Precision AMR Median: {rmsd_metrics['MatP_med']}")

            if self.clearml_logger:
                for k, v in rmsd_metrics_to_log.items():
                    self.clearml_logger.report_scalar(
                        title="RMSD Metrics",
                        series=k,
                        value=v,
                        iteration=0,
                    )

    def validate(self, datamodule):
        trainer = self.get_trainer()
        trainer.validate(
            model=self.model,
            datamodule=datamodule,
        )


def flatten(x):
    if isinstance(x, list):
        for i in x:
            yield from flatten(i)
    else:
        yield x
