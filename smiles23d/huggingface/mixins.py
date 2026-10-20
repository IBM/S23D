from pathlib import Path

from huggingface_hub import PyTorchModelHubMixin
from omegaconf import DictConfig, OmegaConf

from smiles23d.modules.diffusion import DiffusionLM
from smiles23d.tokenizers import PileTokenizer


class S23DDiffusionHF(
    DiffusionLM,
    PyTorchModelHubMixin,
    library_name="smiles23d",
    license="apache-2.0",
    repo_url="https://github.com/IBM/S23D",
    paper_url="https://arxiv.org/abs/2506.19834",
    docs_url="https://github.com/IBM/S23D#readme",
    tags=["chemistry", "molecular-conformer-generation", "diffusion"],
):
    """
    HuggingFace wrapper class for initializing and using the S23D
        Diffusion model with checkpoints from Hugging Face. Adds
        'save_pretrained' and 'push_to_hub' capabilities to the
        DiffusionLM. Can instantiate the DiffusionLM directly from
        Hugging Face with the 'from_pretrained' method.

    Args:
    ----
        *args: Additional arguments passed to the DiffusionLM and
            PyTorchModelHubMixin constructors.
        **kwargs: Keyword arguments passed to the DiffusionLM and
            PyTorchModelHubMixin constructors.

    Returns:
    -------
        S23DDiffusionHF: The initialized S23D Diffusion model.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def save_pretrained(
        self,
        save_directory: str | Path,
        *args,
        config: dict | None = None,
        repo_id: str | None = None,
        push_to_hub: bool = False,
        model_card_kwargs: dict[str, any] | None = None,
        **push_to_hub_kwargs,
    ) -> str | None:
        """
        Save the model to a directory, ready to be pushed to the
            Hugging Face Hub repo. This overrides the default
            save_pretrained method only to create a config dict from
            the Pytorch Lightning Module. It then calls the default
            method.

        Args:
        ----
            save_directory (str | Path): Path to directory in which the
                model weights and configuration will be saved.
            *args: Additional arguments for the save operation.
            config (dict | None): Model configuration specified as a
                key/value dictionary or a dataclass instance.
            repo_id (str | None): ID of the repository on the Hub.
                Used only if `push_to_hub=True`. Will default to the
                folder name if not provided.
            push_to_hub (bool): Whether to push the model to the
                Huggingface Hub after saving it.
            model_card_kwargs (dict[str, any] | None): Additional
                arguments to customize the model card template.
            **push_to_hub_kwargs: Additional key word arguments passed
                to the [`~ModelHubMixin.push_to_hub`] method.

        Returns:
        -------
            str | None: url of the commit on the Hub if
                `push_to_hub=True`, `None` otherwise.
        """
        if not config:
            config = {}
            for k, v in self.hparams.items():
                if isinstance(v, DictConfig):
                    v = OmegaConf.to_object(v)
                config[k] = v

        return super().save_pretrained(
            save_directory,
            *args,
            config=config,
            repo_id=repo_id,
            push_to_hub=push_to_hub,
            model_card_kwargs=model_card_kwargs,
            **push_to_hub_kwargs,
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        map_location="cpu",
        *args,
        force_download: bool = False,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = True,
        **model_kwargs,
    ):
        """
        Download the saved checkpoint from the Huggingface Hub and
            instantiate an instance of the model with that checkpoint.
            Overrides the parent `from_pretrained` class to start a
            Tokenizer, required for instantiating the DiffusionLM model
            but not logged to the checkpoint. Then calls the super
            method, to get the model. Then maps the model to the device
            specified in `map_location`.

        Args:
        ----
            pretrained_model_name_or_path (`str`, `Path`):
                - Either the `model_id` (string) hosted on the Hub.
                - Or a path to a `directory` containing model weights
                    saved using `save_pretrained`
            map_location (`str`, `torch.device`, *optional*): Device the
                model is moved to after loading, such as "cpu" or "cuda".
                Defaults to "cpu".
            revision (`str`, *optional*): Revision of the model on the
                Hub. Can be a branch name, a git tag or any commit id.
                Defaults to the latest commit on `main` branch.
            force_download (`bool`, *optional*): Whether to force
                (re-)downloading the model weights and configuration
                files from the Hub, overriding the existing cache.
                Default: False.
            token (`str` or `bool`, *optional*): The token to use as
                HTTP bearer authorization for remote files. By default,
                it will use the token cached when running
                `hf auth login`.
            cache_dir (`str`, `Path`, *optional*): Path to the folder
                where cached files are stored.
            local_files_only (`bool`, *optional*): If `True`, avoid
                downloading the file and return the path to the local
                cached file if it exists. Defaults to `False`.
            strict (`bool`, *optional*): Whether to raise an error if
                the saved weights do not match the model's state dict
                exactly. Defaults to `True`.
            model_kwargs (`Dict`, *optional*): Additional kwargs to
                pass to the model during initialization.
        """
        tokenizer = PileTokenizer()
        model_kwargs["tokenizer"] = tokenizer
        model = super().from_pretrained(
            pretrained_model_name_or_path,
            *args,
            force_download=force_download,
            token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            revision=revision,
            strict=strict,
            **model_kwargs,
        )
        model.to(map_location)
        return model

    @classmethod
    def _from_pretrained(cls, *, model_id, **kwargs):
        if "diffusion_model" not in kwargs:
            raise FileNotFoundError(
                f"No S23D config.json found for '{model_id}'. Pass a directory "
                "written by save_pretrained or a Hugging Face repo id; for a "
                "private repo, log in with `hf auth login`."
            )
        return super()._from_pretrained(model_id=model_id, **kwargs)
