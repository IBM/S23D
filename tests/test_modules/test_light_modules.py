import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from smiles23d.chem import check_and_add_hs_to_mol
from smiles23d.modules.diffusion import DiffusionLM


def test_diffusion_lm_initialization(diffusion_lm):
    assert isinstance(diffusion_lm, DiffusionLM)


def test_diffusion_lm_train(
    diffusion_lm, geom_datamodule_train, partial_trainer_train_task
):
    max_epochs = 1
    limit_val_batches = 1.0 if geom_datamodule_train.val_dataloader_config else 0.0
    trainer = partial_trainer_train_task(
        max_epochs=max_epochs, limit_val_batches=limit_val_batches
    )
    trainer.fit(diffusion_lm, geom_datamodule_train)
    assert trainer.state.finished


def test_diffusion_lm_test(
    diffusion_lm, geom_datamodule_predict, partial_trainer_predict_task, dict_keys
):
    trainer = partial_trainer_predict_task()
    metrics = trainer.test(diffusion_lm, geom_datamodule_predict)

    assert isinstance(metrics[0], dict)
    assert set(metrics[0].keys()) == dict_keys.metric_keys


def test_diffusion_lm_predict(
    diffusion_lm, geom_datamodule_predict, partial_trainer_predict_task, dict_keys
):
    trainer = partial_trainer_predict_task()
    predictions = trainer.predict(diffusion_lm, geom_datamodule_predict)

    # predictions[0] is the list of molecules in S23D output format
    assert isinstance(predictions[0], list)
    assert isinstance(predictions[0][0]["molecule"], dict)
    assert isinstance(predictions[0][0]["molecule"]["smiles"], str)
    assert isinstance(predictions[0][0]["molecule"]["conformations"], list)
    conf0 = predictions[0][0]["molecule"]["conformations"][0]
    assert isinstance(conf0, dict)
    assert set(conf0.keys()) == dict_keys.predicted_conformation_keys
    assert isinstance(conf0["position"][0], float)
    assert isinstance(conf0["atoms"][0], str)
    assert isinstance(conf0["explicitHs"][0], int)
    assert isinstance(conf0["charges"][0], int)
    assert isinstance(conf0["bonds"][0], list)
    assert isinstance(conf0["bonds"][0][0], int)


def test_diffusion_lm_predict_mol_from_smiles(diffusion_lm, test_values):
    """
    Test predicting Mol objects from SMILES. Output format is a list for
    each SMILES, where each of those lists is a list of generated Mols.
    """
    output_mol = diffusion_lm.predict(test_values.smiles)
    assert isinstance(output_mol, list)
    assert isinstance(output_mol[0], list)
    assert isinstance(output_mol[0][0], Chem.Mol)


def test_diffusion_lm_predict_mol_from_mol(diffusion_lm, test_values):
    """
    Test predicting Mol objects from Mols. Output format is a list for
    each SMILES, where each of those lists is a list of generated Mols.
    """
    input_mol = Chem.MolFromSmiles(test_values.smiles)
    output_mol = diffusion_lm.predict(input_mol)
    assert isinstance(output_mol, list)
    assert isinstance(output_mol[0], list)
    assert isinstance(output_mol[0][0], Chem.Mol)
    assert Chem.MolToSmiles(
        Chem.RemoveHs(output_mol[0][0]), isomericSmiles=False
    ) == Chem.MolToSmiles(input_mol, isomericSmiles=False)


@pytest.mark.parametrize(
    "to_input",
    [str, Chem.MolFromSmiles, lambda smiles: Chem.AddHs(Chem.MolFromSmiles(smiles))],
    ids=["smiles", "mol", "mol_with_hs"],
)
def test_diffusion_lm_predict_adds_hydrogens(diffusion_lm, to_input):
    smiles = "CC(=O)Nc1ccc(O)cc1"
    expected = Chem.AddHs(Chem.MolFromSmiles(smiles))
    output_mol = diffusion_lm.predict(to_input(smiles))
    atoms = [atom.GetSymbol() for atom in output_mol[0][0].GetAtoms()]
    assert atoms == [atom.GetSymbol() for atom in expected.GetAtoms()]


def test_check_and_add_hs_to_mol_places_added_hydrogens():
    mol = Chem.MolFromSmiles("CC(=O)Nc1ccc(O)cc1")
    AllChem.EmbedMolecule(mol, randomSeed=0)
    mol_with_hs = check_and_add_hs_to_mol(mol)
    positions = mol_with_hs.GetConformer().GetPositions()
    assert mol_with_hs.GetNumAtoms() == 20
    assert (abs(positions[mol.GetNumAtoms() :]).sum(axis=1) > 0).all()


def test_diffusion_lm_predict_s23d_from_smiles(diffusion_lm, test_values, dict_keys):
    """
    Test predicting S23D dicts from SMILES. Output format is a dict for
    each SMILES, where each of those dicts is in the S23D output format.
    """
    output_s23d = diffusion_lm.predict(test_values.smiles, output_format="s23d")
    assert isinstance(output_s23d, list)
    assert isinstance(output_s23d[0]["molecule"], dict)
    assert isinstance(output_s23d[0]["molecule"]["smiles"], str)
    assert isinstance(output_s23d[0]["molecule"]["conformations"], list)
    conf0 = output_s23d[0]["molecule"]["conformations"][0]
    assert isinstance(conf0, dict)
    assert set(conf0.keys()) == dict_keys.predicted_conformation_keys
    assert isinstance(conf0["position"][0], float)
    assert isinstance(conf0["atoms"][0], str)
    assert isinstance(conf0["explicitHs"][0], int)
    assert isinstance(conf0["charges"][0], int)
    assert isinstance(conf0["bonds"][0], list)
    assert isinstance(conf0["bonds"][0][0], int)


def test_diffusion_lm_predict_more_molecules_than_batch_size(diffusion_lm, test_values):
    smiles = [
        test_values.smiles,
        "CC(=O)Oc1ccccc1C(=O)O",
        "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    ]
    output_s23d = diffusion_lm.predict(
        smiles, samples_per_molecule=2, batch_size=2, output_format="s23d"
    )
    assert [item["molecule"]["smiles"] for item in output_s23d] == smiles
    assert [len(item["predictions"]) for item in output_s23d] == [2, 2, 2]


@pytest.mark.parametrize(
    ("predict_input", "match"),
    [("C1CC", "Could not create Mol object"), (42, "not supported")],
)
def test_diffusion_lm_predict_invalid_input(diffusion_lm, predict_input, match):
    with pytest.raises(ValueError, match=match):
        diffusion_lm.predict(predict_input)


def test_diffusion_lm_predict_embedding_failure(diffusion_lm, test_values, monkeypatch):
    monkeypatch.setattr(AllChem, "EmbedMolecule", lambda *args, **kwargs: -1)
    with pytest.raises(ValueError, match="Could not embed a conformer"):
        diffusion_lm.predict(test_values.smiles)
