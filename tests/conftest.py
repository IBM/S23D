from functools import partial

import lightning as L
import pytest
from helpers import ConfigPaths, DatamodulePaths, DictKeys, TestValues
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from lightning import Trainer
from omegaconf import OmegaConf

from smiles23d.nexusnet.main import cs  # noqa: F401
from smiles23d.tokenizers import PileTokenizer


def instantiate_module(pars):
    module_pars = OmegaConf.to_container(pars.module, resolve=True)
    module: L.LightningModule = instantiate(
        module_pars, _recursive_=False, _partial_=True
    )
    if pars.tokenizer:
        tokenizer = instantiate(pars.tokenizer)
        module = module(tokenizer=tokenizer)
    else:
        module = module()
    return module


def instantiate_datamodule(pars):
    datamodule: L.LightningDataModule = instantiate(pars.datamodule)
    if pars.tokenizer:
        tokenizer = instantiate(pars.tokenizer)
        datamodule.set_tokenizer(tokenizer)
    return datamodule


def partial_instantiate_trainer(trainer):
    partial_trainer = partial(Trainer, **trainer)
    return partial_trainer


@pytest.fixture
def dict_keys():
    return DictKeys


@pytest.fixture
def test_values():
    return TestValues


@pytest.fixture(scope="session")
def output_dir(tmp_path_factory):
    output_dir = tmp_path_factory.mktemp("outputs")
    return output_dir


@pytest.fixture(scope="session")
def train_paths(output_dir):
    train_paths = {
        "checkpoint": {
            "root_dir": output_dir / "train" / "checkpoints",
            "epoch_dir": output_dir / "train" / "epoch_checkpoints",
        }
    }
    return train_paths


@pytest.fixture(scope="session")
def predict_paths(output_dir):
    save_dir = output_dir / "predict"
    predict_paths = {
        "save_path": save_dir,
        "save_jsonl_file_prefix": save_dir / "output",
        "save_mol_file": save_dir / "output_mols.pkl",
        "save_rmsd_file": save_dir / "output_rmsds.pkl",
    }
    return predict_paths


@pytest.fixture(scope="session")
def huggingface_dir(tmp_path_factory):
    hf_dir = tmp_path_factory.mktemp("hf")
    return hf_dir


@pytest.fixture
def tokenizer():
    return PileTokenizer()


@pytest.fixture(scope="module")
def diffusion_train_config(train_paths):
    with initialize_config_dir(
        config_dir=str(ConfigPaths.config_dir), version_base="1.2"
    ):
        cfg = compose(config_name="test_diffusion_train")
        cfg.datamodule.input_dir = str(DatamodulePaths.input_dir)
        cfg.datamodule.val_dataloader_config.input_dir = str(
            DatamodulePaths.eval_input_dir
        )
        cfg.datamodule.val_dataloader_config.index_dir = str(
            DatamodulePaths.eval_index_dir
        )
        cfg.task.checkpoint.root_dir = str(train_paths["checkpoint"]["root_dir"])
        cfg.task.checkpoint.epoch_dir = str(train_paths["checkpoint"]["epoch_dir"])
    return cfg


@pytest.fixture(scope="module")
def diffusion_predict_config(predict_paths):
    with initialize_config_dir(
        config_dir=str(ConfigPaths.config_dir), version_base="1.2"
    ):
        cfg = compose(config_name="test_diffusion_prediction")
        cfg.datamodule.input_dir = str(DatamodulePaths.eval_input_dir)
        cfg.datamodule.index_dir = str(DatamodulePaths.eval_index_dir)
        cfg.test.save_path = str(predict_paths["save_path"])
    return cfg


@pytest.fixture
def diffusion_lm(diffusion_train_config):
    diffusion_lm = instantiate_module(diffusion_train_config)
    return diffusion_lm


@pytest.fixture
def geom_datamodule_train(diffusion_train_config):
    geom_datamodule_train = instantiate_datamodule(diffusion_train_config)
    return geom_datamodule_train


@pytest.fixture
def geom_datamodule_predict(diffusion_predict_config):
    geom_datamodule_predict = instantiate_datamodule(diffusion_predict_config)
    return geom_datamodule_predict


@pytest.fixture
def partial_trainer_train_task(diffusion_train_config):
    partial_trainer = partial_instantiate_trainer(diffusion_train_config.task.trainer)
    return partial_trainer


@pytest.fixture
def partial_trainer_predict_task(diffusion_predict_config):
    partial_trainer = partial_instantiate_trainer(diffusion_predict_config.test.trainer)
    return partial_trainer


@pytest.fixture
def trained_checkpoint(
    diffusion_lm, geom_datamodule_train, partial_trainer_train_task, train_paths
):
    checkpoint = train_paths["checkpoint"]["root_dir"] / "test.ckpt"
    max_epochs = 1
    limit_val_batches = 1.0 if geom_datamodule_train.val_dataloader_config else 0.0
    trainer = partial_trainer_train_task(
        max_epochs=max_epochs, limit_val_batches=limit_val_batches
    )
    trainer.fit(diffusion_lm, geom_datamodule_train)
    assert trainer.state.finished
    trainer.save_checkpoint(checkpoint)
    return checkpoint
