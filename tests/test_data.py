import pytest
import torch
from hydra.utils import instantiate


def test_nexus_datamodule_train_dl(geom_datamodule_train, tokenizer):
    dm = geom_datamodule_train
    dm.set_tokenizer(tokenizer)
    dm.setup(stage="fit")
    train_dl = dm.train_dataloader()
    sample = next(iter(train_dl))
    assert isinstance(sample["x"], torch.Tensor)


@pytest.mark.parametrize("method", ["test_dataloader", "predict_dataloader"])
def test_inference_dataloader_uses_no_workers(
    diffusion_predict_config, tokenizer, method
):
    dm = instantiate(diffusion_predict_config.datamodule, num_workers=2)
    dm.set_tokenizer(tokenizer)
    with pytest.warns(UserWarning, match="num_workers"):
        dl = getattr(dm, method)()
    assert dl.num_workers == 0
    assert dm.num_workers == 2
