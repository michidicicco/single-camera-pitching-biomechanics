# Reproducibility

This repository is intended to preserve the computations used for the associated study while presenting them through stable, publication-facing filenames and terminology.

The study processing parameters are stored in `config/pipeline_parameters.json`. The computational environment is documented in `config/software_environment.txt`, `requirements.txt`, and `environment.yml`.

## Code migration rule

Renaming a file, reorganizing documentation, or replacing provisional public terminology does not by itself change the method. Changes to formulas, event definitions, processing parameters, feature definitions, model inputs, or model-selection rules are methodological changes and should be documented explicitly.

The public code was regression-checked against the archived study outputs. See `docs/reproducibility_check.md` for the comparison.

## Analytical separation

The relative mechanical and unsupervised clustering pathways remain independent until post hoc integration. Participant identifiers, recording metadata, measurement-QC variables, event-confidence variables, and mechanical-profile outputs are not primary clustering inputs.

## Primary feature matrix

The primary analysis begins with 36 prespecified biomechanical candidate variables. Deterministic missingness, variance, and high-correlation rules yield the 32 retained features used for the associated study. Downstream clustering analyses should consume the model-ready table produced by `prepare_analysis_data.py`, rather than arbitrary tables containing additional numeric fields.

## Release rule

A citable software release should be created only after code compilation, documentation checks, terminology checks, and regression comparisons are satisfactory. The archived release should then be linked to the thesis and associated journal publication through its persistent DOI.
