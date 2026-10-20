from functools import partial

import lightning as L
import torch
from hydra.utils import instantiate
from rdkit import Chem
from rdkit.Chem import Mol

from smiles23d.chem import (
    check_and_add_hs_to_mol,
    embed_rdkit_conformer,
    mol_to_smiles23d,
)
from smiles23d.data.geom import GEOMCollator
from smiles23d.tokenizers.piletok import PileTokenizer
from smiles23d.tools import MolData, RMSDWrapper

from .diffusion_layers import diffusion_factory


class DiffusionLM(L.LightningModule):
    def __init__(
        self,
        diffusion_model,
        score_network,
        optimizer,
        tokenizer,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["tokenizer"])
        self.optimizer_options = optimizer
        self.diffusion_model = instantiate(diffusion_model)
        self.score_network = instantiate(score_network, _partial_=True)
        self.set_tokenizer(tokenizer)
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.predict_step_outputs = []
        self.n_steps = 200
        self.n_cpus = 1

    def set_tokenizer(self, tokenizer):
        self.score_network = self.score_network(vocab_size=tokenizer.n_tokens)
        self.vocab = tokenizer.reverse_vocab

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        (
            attention_mask_aux_tokens,
            attention_mask,
            padding_mask,
        ) = self.score_network.create_masks(batch["atoms"], batch["short_paths"])

        score_network = partial(
            self.score_network,
            batch,
            attention_mask,
            attention_mask_aux_tokens,
        )
        loss = self.diffusion_model.loss(batch["x"], padding_mask, score_network)
        self.log("train_loss", loss.item(), on_epoch=True, on_step=True, prog_bar=True)
        return loss

    def set_predict_parameters(self, **diffusion_kwargs):
        """
        Set the sampler and the number of diffusion steps used by `predict`.

        The new sampler keeps the beta schedule, scale and score settings of the
        current diffusion model. Without a call to this method, `predict` uses the
        diffusion model from the model config with 200 steps.

        Args:
        ----
            **diffusion_kwargs: The keyword arguments `kind` and `n_steps`,
                both required. Other keys are ignored.

                - kind (str): "VPSDE" for the reverse SDE sampler
                  (CoMVPSDE) or "DDIM" for the DDIM sampler.
                - n_steps (int): Number of diffusion steps used for sampling.

        Raises:
        ------
            KeyError: If `n_steps` is missing.
            ValueError: If `kind` is missing or is not "VPSDE" or "DDIM".
        """
        self.diffusion_model, n_steps = diffusion_factory(
            self.diffusion_model, **diffusion_kwargs
        )
        self.n_steps = n_steps

    @torch.no_grad()
    def _sample(self, batch):
        batch, molecules = batch
        atoms = batch["atoms"]
        molecules = molecules.value
        samples_per_molecule = len(atoms) // len(molecules)

        atoms = batch["atoms"]

        (
            attention_mask_aux_tokens,
            attention_mask,
            padding_mask,
        ) = self.score_network.create_masks(atoms, batch["short_paths"])

        graph_encoding = self.score_network.encode_graph(batch, attention_mask)

        score_network = partial(
            self.score_network.decode_structure,
            graph_encoding,
            batch,
            attention_mask_aux_tokens,
        )

        predict_x = self.diffusion_model.reverse(
            score_network,
            padding_mask,
            device=self.device,
            n_steps=self.n_steps,
        )

        items = []
        for index, mol in enumerate(molecules):
            predictions = []
            for i in range(samples_per_molecule):
                predictions.append(
                    predict_x[index * samples_per_molecule + i]
                    .cpu()
                    .numpy()[
                        : padding_mask[index * samples_per_molecule + i].sum().int()
                    ]
                    .tolist()
                )
            item = {"molecule": mol, "predictions": predictions}
            items.append(item)
        return items

    def on_validation_epoch_start(self):
        self.validation_step_outputs = []

    def validation_step(self, batch, batch_idx):
        samples = self._sample(batch)
        self.validation_step_outputs.extend(samples)

    def on_validation_epoch_end(self):
        mols = self.validation_step_outputs

        # Construct Mol objects
        mol_data = MolData(n_cpus=self.n_cpus)
        mol_data.load_molecules(mols)

        # Run RMSD metrics
        rmsd_wrapper = RMSDWrapper(mol_data=mol_data, n_cpus=self.n_cpus)
        rmsd_metrics = rmsd_wrapper()

        self.log_dict(rmsd_metrics, on_epoch=True, prog_bar=True)
        self.validation_step_outputs.clear()  # free memory

    def on_test_epoch_start(self):
        self.test_step_outputs = []

    def test_step(self, batch, batch_idx):
        samples = self._sample(batch)
        self.test_step_outputs.extend(samples)

    def on_test_epoch_end(self):
        mols = self.test_step_outputs

        # Construct Mol objects
        mol_data = MolData(n_cpus=self.n_cpus)
        mol_data.load_molecules(mols)

        # Run RMSD metrics
        rmsd_wrapper = RMSDWrapper(mol_data=mol_data, n_cpus=self.n_cpus)
        rmsd_metrics = rmsd_wrapper()

        self.log_dict(rmsd_metrics, on_epoch=True, prog_bar=True)
        self.test_step_outputs.clear()

    def predict_step(self, batch, batch_idx):
        samples = self._sample(batch)
        return samples

    def configure_optimizers(self) -> torch.optim.AdamW:
        """
        Pytorch lightning function to define optimizers and LR schedulers.

        Returns
        -------
            torch.optim.AdamW: AdamW optimizer over the parameter groups
                from `get_weight_decay_groups`.
        """
        decay = self.optimizer_options.get("weight_decay", None)
        optimizer_options = self.optimizer_options.copy()
        if decay is not None:
            del optimizer_options["weight_decay"]
        par_groups = get_weight_decay_groups(decay, self.score_network)
        optimizer = torch.optim.AdamW(**optimizer_options, params=par_groups)
        return optimizer

    def on_save_checkpoint(self, checkpoint):
        checkpoint[
            "class_path"
        ] = f"{self.__class__.__module__}.{self.__class__.__name__}"

    def predict(
        self,
        predict_input: str | Mol | list,
        samples_per_molecule: int = 1,
        batch_size: int = 128,
        output_format: str = "mol",
    ):
        """
        Generate 3D conformers for one or more molecules.

        Hydrogens are always added: the hydrogens an input leaves implicit are
        added as explicit atoms, and coordinates are generated for every atom. An
        input without a conformer first gets one from RDKit ETKDG. Each input
        should be one connected molecule; inputs with several fragments, such as
        salts, are not rejected, but the models were trained on connected
        molecules only, and some multi-fragment inputs fail. The supported elements are the element symbols in the PileTokenizer
        vocabulary: B, Bi, Br, C, Cl, F, H, I, N, O, P, S and Si. Atoms of any
        other element are encoded as the MASK token, without a warning. The
        sampler (`kind`, "VPSDE" or "DDIM") and the number of diffusion steps
        (`n_steps`) are set with `set_predict_parameters`; without it, the
        diffusion model from the model config is used with 200 steps.

        Args:
        ----
            predict_input (str | Mol | list): A SMILES string, an RDKit Mol, or
                a list of them.
            samples_per_molecule (int): Number of conformers to generate for
                each input. Defaults to 1.
            batch_size (int): Number of input molecules per forward pass, each
                repeated `samples_per_molecule` times. Defaults to 128.
            output_format (str): "mol" or "s23d". Defaults to "mol".

        Returns:
        -------
            list: With "mol", one list of RDKit Mol objects per distinct input
                SMILES, in input order, each Mol holding one generated
                conformer; repeated inputs are merged into one entry. With
                "s23d", one dict per input with the S23D `molecule` and its
                `predictions`, a list of atom coordinates per sample.

        Raises:
        ------
            ValueError: If an input is neither a SMILES string nor an RDKit Mol,
                a SMILES string cannot be parsed, RDKit cannot add hydrogens or
                embed a conformer, or `output_format` is not "mol" or "s23d".
        """
        tokenizer = PileTokenizer()
        collator = GEOMCollator(
            tokenizer=tokenizer,
            num_workers=1,
            samples_per_molecule=samples_per_molecule,
        )
        collator.setup_rng(42)

        if not isinstance(predict_input, list):
            predict_input = [predict_input]

        batch_data = []

        # Create a list of S23D molecule objects to turn into batches
        for molecule in predict_input:
            if isinstance(molecule, str):
                smiles = molecule
                m = Chem.MolFromSmiles(smiles)
                if m is None:
                    raise ValueError(f"Could not create Mol object for: {molecule}")
            elif isinstance(molecule, Mol):
                smiles = Chem.MolToSmiles(molecule)
                m = molecule
            else:
                raise ValueError(f"molecule type not supported yet: {molecule}")

            m = check_and_add_hs_to_mol(m)
            m = embed_rdkit_conformer(m)

            s23d_mol = mol_to_smiles23d(m)

            batch_data.append(
                {"smiles": smiles, collator.conformations_field: [s23d_mol]}
            )

        batches = []
        for i in range(0, len(batch_data), batch_size):
            batches.append(batch_data[i : i + batch_size])

        samples = []
        for batch in batches:
            batch, mols = collator.predict_collator(batch=batch)
            batch = {key: item.to(self.device) for key, item in batch.items()}
            batch = (batch, mols)
            samples.extend(self._sample(batch))

        if output_format == "mol":
            mol_data = MolData(n_cpus=self.n_cpus)
            mol_data.load_molecules(samples)
            output_molecules = mol_data.mol_pred
        elif output_format == "s23d":
            output_molecules = samples
        else:
            raise ValueError(f"output_format '{output_format}' not supported")

        return output_molecules


def get_weight_decay_groups(weight_decay, model):
    """
    Code from MinGPT
    https://github.com/karpathy/minGPT/blob/master/mingpt/model.py.
    """
    if weight_decay is not None:
        decay = set()
        no_decay = set()

        for mn, m in model.named_parameters():
            if mn.endswith("bias") or mn.endswith("basis") or mn.endswith("scale"):
                no_decay.add(mn)
            elif mn.endswith("weight") and "embedding" not in mn:
                decay.add(mn)
            else:
                no_decay.add(mn)

        # validate that we considered every parameter
        param_dict = dict(model.named_parameters())
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), f"parameters {str(inter_params)} made it into both decay/no_decay sets!"
        assert (
            len(param_dict.keys() - union_params) == 0
        ), f"parameters {str(param_dict.keys() - union_params)} were not separated into either decay/no_decay set!"

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(no_decay)],
                "weight_decay": 0.0,
            },
        ]
    else:
        optim_groups = [
            {
                "params": [p for n, p in model.named_parameters()],
                "weight_decay": 0.0,
            },
        ]

    return optim_groups
