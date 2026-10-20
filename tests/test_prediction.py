import pickle

import lightning as L
import pytest
from hydra.utils import instantiate
from lightning.pytorch.callbacks import Callback
from rdkit import Chem

from smiles23d.prediction.diffusion import S23DPredict
from smiles23d.tools import MolData


class BatchIndexModule(L.LightningModule):
    def predict_step(self, batch, batch_idx):
        return batch_idx


class InterruptAfterFirstBatch(Callback):
    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        raise RuntimeError("interrupted")


@pytest.fixture
def partial_s23d_predict(diffusion_predict_config, trained_checkpoint):
    diffusion_predict_config.test.checkpoint = str(trained_checkpoint)
    tester = instantiate(diffusion_predict_config.test, _partial_=True)
    return tester


def test_s23d_predict(
    partial_s23d_predict,
    geom_datamodule_predict,
    tokenizer,
    predict_paths,
    dict_keys,
):
    tester = partial_s23d_predict(tokenizer=tokenizer)
    tester.load_checkpoint()
    tester.test(geom_datamodule_predict)

    mol_data = pickle.load(open(predict_paths["save_mol_file"], "rb"))
    rmsd_metrics = pickle.load(open(predict_paths["save_rmsd_file"], "rb"))

    assert isinstance(mol_data, MolData)
    assert isinstance(mol_data.mol_true[0], Chem.Mol)
    assert isinstance(mol_data.mol_pred[0], list)
    assert isinstance(mol_data.mol_pred[0][0], Chem.Mol)
    assert isinstance(mol_data.smiles[0], str)
    assert isinstance(rmsd_metrics, dict)
    assert set(rmsd_metrics.keys()) == dict_keys.rmsd_file_keys

    # Run again with remove output flag
    tester2 = partial_s23d_predict(tokenizer=tokenizer, remove_output=True)
    tester2.load_checkpoint()
    tester2.test(geom_datamodule_predict)

    mol_data = pickle.load(open(predict_paths["save_mol_file"], "rb"))
    rmsd_metrics = pickle.load(open(predict_paths["save_rmsd_file"], "rb"))

    assert isinstance(mol_data, MolData)
    assert isinstance(mol_data.mol_true[0], Chem.Mol)
    assert isinstance(mol_data.mol_pred[0], list)
    assert isinstance(mol_data.mol_pred[0][0], Chem.Mol)
    assert isinstance(mol_data.smiles[0], str)
    assert isinstance(rmsd_metrics, dict)
    assert set(rmsd_metrics.keys()) == dict_keys.rmsd_file_keys


def test_s23d_predict_save_dataloader_state(
    partial_s23d_predict,
    geom_datamodule_predict,
    tokenizer,
    predict_paths,
    dict_keys,
):
    tester = partial_s23d_predict(tokenizer=tokenizer, save_dataloader_state=True)
    tester.load_checkpoint()
    tester.test(geom_datamodule_predict)

    mol_data = pickle.load(open(predict_paths["save_mol_file"], "rb"))
    rmsd_metrics = pickle.load(open(predict_paths["save_rmsd_file"], "rb"))

    assert isinstance(mol_data, MolData)
    assert isinstance(mol_data.mol_true[0], Chem.Mol)
    assert isinstance(mol_data.mol_pred[0], list)
    assert isinstance(mol_data.mol_pred[0][0], Chem.Mol)
    assert isinstance(mol_data.smiles[0], str)
    assert isinstance(rmsd_metrics, dict)
    assert set(rmsd_metrics.keys()) == dict_keys.rmsd_file_keys


def test_dataloader_state_callback_resumes_and_cleans_up(
    tmp_path, monkeypatch, diffusion_predict_config, tokenizer
):
    monkeypatch.chdir(tmp_path)
    save_path = tmp_path / "output"
    save_path.mkdir()
    state_file = save_path / "nexus_test_dataloader_state_0.pt"
    datamodule = instantiate(diffusion_predict_config.datamodule, batch_size=4)
    datamodule.set_tokenizer(tokenizer)

    def get_trainer(*callbacks):
        trainer = S23DPredict(
            checkpoint=None,
            tokenizer=tokenizer,
            diffusion_kwargs={},
            save_path=save_path,
            save_jsonl_file_prefix=None,
            trainer={
                "accelerator": "cpu",
                "devices": 1,
                "logger": False,
                "enable_progress_bar": False,
                "enable_model_summary": False,
            },
            save_dataloader_state=True,
        ).get_trainer()
        trainer.callbacks.extend(callbacks)
        return trainer

    with pytest.raises(RuntimeError, match="interrupted"):
        get_trainer(InterruptAfterFirstBatch()).predict(
            BatchIndexModule(), datamodule=datamodule
        )
    assert state_file.exists()

    predictions = get_trainer().predict(BatchIndexModule(), datamodule=datamodule)
    assert len(predictions) == 3
    assert not state_file.exists()
    assert not list(tmp_path.glob("nexus_test_dataloader_state*"))
