# Reproducibility

The public repository should reproduce the validated study workflow without silently changing the computations that generated the thesis and manuscript results.

The validated processing parameters are stored in `config/pipeline_parameters.json`. The study environment is recorded in `config/software_environment.txt`, `requirements.txt`, and `environment.yml`.

## Code Migration Rule

The public script names intentionally omit development version suffixes and provisional adjectives. Renaming a file is acceptable; changing its computations is a methodological change.

Before the first archived software release:

1. copy each canonical study script into its neutral public filename;
2. update imports, script references, and output paths without changing formulas or validated parameters;
3. run the public pipeline on the same test input used for the archived study workflow;
4. compare event frames, retained feature values, QC assignments, analysis matrices, PCA outputs, cluster assignments, and study summaries;
5. document any intentional difference;
6. create the public release only after the regression comparison is satisfactory.

The relative mechanical and unsupervised clustering pathways must remain independent until post hoc integration. Participant identifiers, recording metadata, measurement-QC variables, event-confidence variables, and mechanical-profile outputs must not enter the primary clustering feature matrix.
