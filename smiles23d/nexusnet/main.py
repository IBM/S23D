from dataclasses import dataclass

import hydra
import lightning as L
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import OmegaConf

from .logbook import start_clearml_logger


@dataclass
class MainSchema:
    datamodule: dict
    module: dict | None = None
    test: dict | None = None
    task: dict | None = None
    clearml: dict | None = None
    tokenizer: dict | None = None
    vars: dict | None = None


default_config_name = "config"

cs = ConfigStore.instance()
cs.store(name="nexusnet_config", node=MainSchema)


@hydra.main(config_path=".", config_name=default_config_name, version_base="1.2")
def main(pars: MainSchema) -> None:
    if pars.clearml:
        start_clearml_logger(**pars.clearml)

    datamodule: L.LightningDataModule = instantiate(pars.datamodule)
    if pars.tokenizer:
        tokenizer = instantiate(pars.tokenizer)
        datamodule.set_tokenizer(tokenizer)

    module_pars = OmegaConf.to_container(pars.module, resolve=True)
    module: L.LightningModule = instantiate(
        module_pars, _recursive_=False, _partial_=True
    )
    if pars.tokenizer:
        module = module(tokenizer=tokenizer)
    else:
        module = module()

    if not pars.task:
        raise ValueError("Training config should have task parameters.")

    task = instantiate(pars.task, _partial_=True)
    task = task(datamodule=datamodule, module=module)
    fit = task.fit
    fit(module, datamodule=datamodule)


@hydra.main(config_path=".", config_name=default_config_name, version_base="1.2")
def test(pars: MainSchema) -> None:
    if pars.clearml:
        clearml_logger = start_clearml_logger(**pars.clearml)
    else:
        clearml_logger = None

    datamodule: L.LightningDataModule = instantiate(pars.datamodule)
    tokenizer = instantiate(pars.tokenizer)
    datamodule.set_tokenizer(tokenizer)

    tester = instantiate(pars.test, _partial_=True)
    tester = tester(tokenizer=tokenizer, clearml_logger=clearml_logger)
    tester.load_checkpoint()
    tester.test(datamodule)


if __name__ == "__main__":
    main()
