import importlib
import os
import shutil

import pytest
import torch
from safetensors.torch import load_file, save_file

from smiles23d.huggingface import S23DDiffusionHF
from smiles23d.prediction import S23DPredict


# From https://github.com/BiomedSciAI/biomed-multi-view/blob/main/bmfm_sm/api/tests/test_smmv_api.py
def all_match(model1, model2):
    all_match = True
    for sd1, sd2 in zip(model1.state_dict().items(), model2.state_dict().items()):
        if not torch.equal(sd1[1], sd2[1]):
            all_match = False
    return all_match


@pytest.fixture
def model_from_cp(tokenizer, trained_checkpoint):
    checkpoint_cp = torch.load(trained_checkpoint, weights_only=False)
    module_path, class_name = checkpoint_cp["class_path"].rsplit(".", 1)
    module = importlib.import_module(module_path)
    ModelClass = getattr(module, class_name)
    model = ModelClass.load_from_checkpoint(
        trained_checkpoint, tokenizer=tokenizer, weights_only=False
    )
    return model


@pytest.fixture
def huggingface_path(huggingface_dir):
    return huggingface_dir / "hf"


@pytest.fixture
def hf_model_from_cp(huggingface_path, trained_checkpoint, tokenizer):
    model = S23DDiffusionHF.load_from_checkpoint(
        trained_checkpoint, tokenizer=tokenizer, weights_only=False
    )
    model.save_pretrained(huggingface_path)
    return model


@pytest.fixture
def model_from_hf(huggingface_path, hf_model_from_cp):
    model = S23DDiffusionHF.from_pretrained(
        huggingface_path, map_location=hf_model_from_cp.device
    )
    return model


def test_consistent_loading_checkpoint_huggingface(
    model_from_cp, hf_model_from_cp, model_from_hf
):
    assert all_match(model_from_cp, hf_model_from_cp)
    assert all_match(model_from_cp, model_from_hf)
    assert all_match(hf_model_from_cp, model_from_hf)


def test_save_pretrained_huggingface(tmp_path, model_from_hf):
    save_dir = tmp_path / "hf"
    model_from_hf.save_pretrained(save_dir)
    assert "model.safetensors" in os.listdir(save_dir)
    assert "README.md" in os.listdir(save_dir)
    assert "config.json" in os.listdir(save_dir)
    card = (save_dir / "README.md").read_text()
    assert "https://github.com/IBM/S23D" in card
    assert "[More Information Needed]" not in card


def test_from_pretrained_missing_weights(tmp_path, huggingface_path, hf_model_from_cp):
    state_dict = load_file(huggingface_path / "model.safetensors")
    missing_key = sorted(state_dict)[0]
    del state_dict[missing_key]
    partial_path = tmp_path / "partial"
    partial_path.mkdir()
    save_file(state_dict, partial_path / "model.safetensors")
    shutil.copy(huggingface_path / "config.json", partial_path)
    with pytest.raises(RuntimeError, match="Missing key"):
        S23DDiffusionHF.from_pretrained(partial_path)


def test_from_pretrained_missing_config(tmp_path):
    with pytest.raises(FileNotFoundError, match="No S23D config.json"):
        S23DDiffusionHF.from_pretrained(tmp_path)


def test_predict_loads_huggingface(huggingface_path, hf_model_from_cp, tokenizer):
    predictor = S23DPredict(
        checkpoint=str(huggingface_path),
        tokenizer=tokenizer,
        diffusion_kwargs={"kind": "VPSDE", "n_steps": 300},
    )
    predictor.load_checkpoint()
    assert all_match(hf_model_from_cp.score_network, predictor.model.score_network)
    assert predictor.model.n_steps == 300
