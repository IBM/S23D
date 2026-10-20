import os


def test_nexusnet_main(
    tmp_path,
    diffusion_train_config,
    diffusion_predict_config,
):
    from smiles23d.nexusnet.main import main

    root_dir = tmp_path / "checkpoints"
    epoch_dir = tmp_path / "epoch_checkpoints"
    diffusion_train_config.task.checkpoint.root_dir = root_dir
    diffusion_train_config.task.checkpoint.epoch_dir = epoch_dir
    diffusion_train_config.task.trainer.max_epochs = 2
    main(diffusion_train_config)

    assert "last.ckpt" in os.listdir(root_dir)
    assert "s23d_test_epoch_0.ckpt" in os.listdir(epoch_dir)
    assert "s23d_test_epoch_1.ckpt" in os.listdir(epoch_dir)

    from smiles23d.nexusnet.main import test

    checkpoint = root_dir / "last.ckpt"
    save_path = tmp_path / "output"
    diffusion_predict_config.test.checkpoint = checkpoint
    diffusion_predict_config.test.save_path = save_path
    test(diffusion_predict_config)

    assert "output_0.jsonl" in os.listdir(save_path)
    assert "output_mols.pkl" in os.listdir(save_path)
    assert "output_rmsds.pkl" in os.listdir(save_path)
