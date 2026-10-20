from pathlib import Path

RESOURCE_ROOT = Path(__file__).parent / "resources"


class DatamodulePaths:
    input_dir = RESOURCE_ROOT / "geom_drugs_tinydb"
    eval_input_dir = input_dir / "zarr"
    eval_index_dir = input_dir / "index" / "eval_index"


class ConfigPaths:
    config_dir = RESOURCE_ROOT / "configs"
    release_config_dir = Path(__file__).parent.parent / "configs"


class DictKeys:
    metric_keys = {
        "CovR_mean",
        "CovR_med",
        "CovP_mean",
        "CovP_med",
        "MatR_mean",
        "MatR_med",
        "MatP_mean",
        "MatP_med",
    }
    rmsd_file_keys = metric_keys | {"all_mol_rmsds"}
    predicted_conformation_keys = {
        "position",
        "atoms",
        "explicitHs",
        "charges",
        "bonds",
    }


class TestValues:
    smiles = "[H]c1s/c(=N/S(=O)(=O)c2c([H])c([H])c(C([H])([H])[H])c([H])c2[H])n(C([H])([H])[H])c1[H]"
