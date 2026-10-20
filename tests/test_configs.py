import pytest
from helpers import ConfigPaths
from hydra import compose, initialize_config_dir
from hydra.utils import get_class
from omegaconf import OmegaConf

TRAIN_CONFIGS = ["s23d_s_stage1", "s23d_s_stage2", "s23d_b_stage1", "s23d_b_stage2"]
EVAL_CONFIGS = ["s23d_s_eval2k", "s23d_b_eval2k"]


def compose_release_config(name):
    with initialize_config_dir(
        config_dir=str(ConfigPaths.release_config_dir), version_base="1.2"
    ):
        cfg = compose(config_name=name)
    return cfg


def iter_targets(node):
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "_target_":
                yield value
            else:
                yield from iter_targets(value)
    elif isinstance(node, list):
        for value in node:
            yield from iter_targets(value)


@pytest.fixture(autouse=True)
def data_root(monkeypatch, tmp_path):
    monkeypatch.setenv("SMILES23D_DATA", str(tmp_path))
    return tmp_path


def test_release_config_names():
    names = {path.stem for path in ConfigPaths.release_config_dir.glob("*.yaml")}
    assert names == set(TRAIN_CONFIGS + EVAL_CONFIGS)


@pytest.mark.parametrize("name", TRAIN_CONFIGS + EVAL_CONFIGS)
def test_targets_resolve(name):
    cfg = compose_release_config(name)
    targets = list(iter_targets(OmegaConf.to_container(cfg, resolve=False)))
    assert targets
    for target in targets:
        get_class(target)


@pytest.mark.parametrize("name", TRAIN_CONFIGS)
def test_train_config(name, data_root):
    cfg = compose_release_config(name)
    stage2 = name.endswith("stage2")
    assert cfg.datamodule.input_dir == str(data_root)
    assert cfg.datamodule.val_dataloader_config is None
    assert cfg.datamodule.removeHs is not stage2
    if stage2:
        assert cfg.task.checkpoint.load_datamodule_state is False
    else:
        assert "load_datamodule_state" not in cfg.task.checkpoint
    assert "limit_val_batches" not in cfg.task.trainer
    assert "devices" not in cfg.task.trainer
    assert cfg.clearml is None


@pytest.mark.parametrize("name", EVAL_CONFIGS)
def test_eval_config(name, data_root):
    cfg = compose_release_config(name)
    datamodule = cfg.datamodule
    assert datamodule.input_dir == str(data_root / "zarr")
    assert datamodule.index_dir == str(data_root / "index" / "eval_index")
    assert datamodule.predict_split == "test_1000"
    assert datamodule.start_idx == 0
    assert datamodule.end_idx == 208304
    assert datamodule.samples_per_molecule == 1
    assert datamodule.num_workers == 0
    assert datamodule.removeHs is False
    assert cfg.test.trainer.devices == 1
    assert OmegaConf.is_missing(cfg.test, "checkpoint")


def test_eval_configs_differ_only_in_save_path():
    s_cfg, b_cfg = (
        OmegaConf.to_container(compose_release_config(name)) for name in EVAL_CONFIGS
    )
    assert s_cfg["test"].pop("save_path") != b_cfg["test"].pop("save_path")
    assert s_cfg == b_cfg
