# Domain-Coref

Domain-Coref is a multi-domain Chinese identity-coreference resource
covering personal pronouns, demonstratives, definite noun phrases,
zero anaphora, and event reference.

This repository provides the custom code used to generate, validate,
convert, split, and package the Domain-Coref dataset.

## Code structure

- `generation/`: candidate-data generation utilities.
- `validation/`: corpus integrity and annotation validation.
- `conversion/`: conversion to the released representations.
- `splitting/`: fixed document split utilities.
- `release/`: construction and validation of the public release.

## Data

The frozen Domain-Coref dataset is distributed separately through Zenodo.
