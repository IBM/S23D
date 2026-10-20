# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

## [1.0.0]

### Added
- Initial public release of S23D.
- Pretrained S23D-S and S23D-B weights on Hugging Face, loaded with
  `S23DDiffusionHF.from_pretrained`.
- `predict()`, which generates 3D conformers, with hydrogens, from SMILES strings
  or RDKit molecules.
- Configs for the two training stages of S23D-S and S23D-B, pretraining without
  hydrogens and fine-tuning with explicit hydrogens, run with `s23dtrain`.
- GEOM-DRUGS evaluation configs for `test_1000`, run with `s23dtest`, which
  generate 2K conformers for a molecule with K reference conformers.
- GEOM-DRUGS preprocessing scripts that build the `drugs_20` dataset from the
  MCF archives.
- The prebuilt GEOM-DRUGS data used for training and evaluation, on Hugging
  Face.

[unreleased]: https://github.com/IBM/S23D/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/IBM/S23D/releases/tag/v1.0.0
