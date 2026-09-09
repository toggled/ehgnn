# EdgeMask-HGNN

This package contains the source, fixed experiment specifications, and aggregation utilities needed to reproduce the experiments in the paper: **Supervised Incidence Sparsification of Hypergraphs under Budget Constraints**

It contains the code and fixed settings for the datasets, methods, and
analyses reported in the LoG paper. The selected hyperparameter values are
included so that the reported evaluations can be run directly.

The reported node-classification
runs were tested on Python 3.9, PyTorch 2.0.0 with CUDA 11.8, and a 32 GB NVIDIA V100.

Run all commands from the package root. Paths beginning with `./` are relative
to this directory.

## Environment

```bash
conda create -n edgemask-hgnn \
  python=3.9 pytorch==2.0.0 pytorch-cuda=11.8 \
  cuda-nvrtc-dev=11.8.89 mkl=2023.1.0 \
  -c pytorch -c nvidia -y
conda activate edgemask-hgnn

python -m pip install \
  torch_scatter==2.1.2 torch_sparse==0.6.18 \
  -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
python -m pip install torch-geometric==2.6.1
python -m pip install -r requirements.txt
```

## Data

Download `data.zip` from the project
[data link](https://drive.google.com/open?id=1-wyCTVXXMvxieJvFR_9-G5w3c8dTfuSG&usp=drive_fs).
The linked archive is 236.7 MiB; extracting every dataset in it uses 2.19 GiB.
Only 23 files, totaling 121.8 MiB, are needed for this paper. Extract just
those files with:

```bash
python scripts/extract_node_data.py ~/Downloads/data.zip
```

The expected archive SHA-256 is
`092271df6841226eed6da31dc84ac463b64379ed5a40ff4b11885b44cc448f50`.
The extraction command verifies this checksum and creates these paths:

```text
data/hetero/{actor,twitch,pokec}/
data/AllSet_all_raw_data/yelp/
data/AllSet_all_raw_data/coauthorship/dblp/
data/AllSet_all_raw_data/walmart-trips/
data/AllSet_all_raw_data/cocitation/cora/
```

The remaining files in the archive are not used by the LoG experiments.

For the hyperedge-prediction experiment, place the original Hyper-SAGNN
`train_data.npz` and `test_data.npz` files under:

```text
Hyper-SAGNN-master/data/{wordnet,drug,MovieLens}/
```

Check the required files before running experiments:

```bash
python check_data.py --scope all
python dataset_statistics.py
```

`dataset_statistics.py` regenerates the counts and hyperedge-homophily values
reported in Table 1.

## Run One Model

`run_model.py` runs one seed using the paper's split, hyperparameters,
early-stopping rule, exact budget, and fixed self-loop convention. Results are
written to `outputs/single_runs/` unless `--output` is supplied.

```bash
# EHGNN-F
python run_model.py --model ehgnn-f --dataset actor --budget 0.5 --seed 0

# Exact-budget controls
python run_model.py --model random-fixed --dataset actor --budget 0.5 --seed 0
python run_model.py --model random-resampled --dataset actor --budget 0.5 --seed 0
python run_model.py --model cardinality --dataset actor --budget 0.5 --seed 0
python run_model.py --model laplacian-proxy --dataset actor --budget 0.5 --seed 0

# Full-retention and feature-only references
python run_model.py --model full --dataset actor --seed 0
python run_model.py --model mlp --dataset actor --seed 0
python run_model.py --model majority --dataset actor --seed 0 --device cpu
```

Valid node-classification datasets are `actor`, `twitch`, `pokec`, `yelp`,
`coauthor_dblp`, and `walmart-trips`. Valid sparse budgets are `0.1`, `0.2`,
`0.3`, and `0.5`.

## Main Tables

Run all cells for Tables 2--4 with:

```bash
DEVICE=cuda:0 bash scripts/run_core_tables.sh
```

The script writes the table inputs to:

| Result | Output |
|---|---|
| Table 2, Actor--Yelp | `./outputs/tables/table_2/actor_through_yelp.csv` |
| Table 2, DBLP-CA and Walmart | `./outputs/tables/table_2/dblp_ca_and_walmart.csv` |
| Table 2, six-dataset tests | `./outputs/tables/table_2/six_dataset_paired_comparisons.csv` |
| Table 3, Random-Resampled | `./outputs/tables/table_3/method_summary.csv` |
| Table 4, MLP and majority | `./outputs/tables/table_4/method_summary.csv` |

`python run_experiment.py --help` lists the supplementary analyses available
through the common entry point.

After the core runs, regenerate the score and mask analyses with:

```bash
python summarize_mask_movement.py
python run_experiment.py retained-structure
```

These commands produce the score/ranking, effective-density, and retained
multi-node-structure summaries from the saved masks.

## Additional Models

### HSL and HERALD (Table 7)

The package does not redistribute these repositories. Fetch the exact upstream
revisions and apply the documented HSL forward-mask correction, then run their
validation selection and ten-seed evaluations:

```bash
bash scripts/fetch_third_party.sh

python run_experiment.py hsl tune --device cuda:0
python run_experiment.py hsl freeze-selection
python run_experiment.py hsl evaluate --device cuda:0
python run_experiment.py hsl-summary

python run_experiment.py herald tune --device cuda:0
python run_experiment.py herald freeze-selection
python run_experiment.py herald evaluate --device cuda:0
python run_experiment.py task-aware-summary
python collect_outputs.py
```

The final collection step writes the combined task-aware results to
`./outputs/tables/table_7/task_aware_baselines.csv`.

### Alternative Scorers (Table 15)

Run the main-table workflow first; its EHGNN-F and Random-Fixed records are
the references used in this summary.

```bash
python conditioned_scorer_study.py evaluate --variants f_cond f_cond_lr --device cuda:0
python conditioned_scorer_study.py summarize --variants f_cond f_cond_lr
```

This reproduces EHGNN-F, the low-rank conditioned scorer, the direct
full-matrix feasibility outcomes, and their Random-Fixed reference using the
selected configuration bundled in `experiment_specs/conditioned_scorer_selection.json`.
To repeat validation selection itself, run `tune` followed by `select`, then
pass the generated `.work/conditioned_scorers/frozen_config.json` with
`--frozen-config` during evaluation and summarization.

### AllSetTransformer and ED-HNN (Table 16)

```bash
python backbone_study.py --device cuda:0
python backbone_study.py --summarize
```

Use `--backbones`, `--datasets`, `--methods`, and `--seeds` to run an individual
cell. The method names in the specification are `Full`, `Random-Fixed`, and
`EHGNN-F`.

```bash
python backbone_study.py \
  --backbones AllSetTransformer --datasets actor \
  --methods EHGNN-F --seeds 0 --device cuda:0
```

### Hyper-SAGNN Hyperedge Prediction (Table 17)

```bash
python hyperedge_prediction.py --device cuda:0
python hyperedge_prediction.py --summarize
```

For one cell, for example:

```bash
python hyperedge_prediction.py \
  --datasets wordnet --methods EHGNN-F --seeds 0 --device cuda:0
```

Valid methods are `Hyper-SAGNN`, `FullContext`, `Random-Fixed`, and `EHGNN-F`.

## Other Reported Analyses

The following commands reproduce the remaining experimental and theory
appendix results. Each command uses its corresponding fixed experiment
specification by default.

For scripts that accept it, `--stage` selects the experiment phase, such as
preparation, evaluation, or aggregation, so completed phases need not be rerun.

```bash
# Table 8: robustness over 15 independently generated splits
python run_experiment.py split-robustness audit-splits
python run_experiment.py split-robustness evaluate --device cuda:0
python run_experiment.py split-robustness summarize

# Table 9: connected and isolated test-node analysis
python run_experiment.py connected-nodes run --device cuda:0
python run_experiment.py connected-nodes summarize

# Selected-gradient exposure reported with Table 12
python run_experiment.py selected-gradients run --device cuda:0
python run_experiment.py selected-gradients aggregate

# Tables 13--14 and Figure 2(a): controlled corruption
python run_experiment.py controlled-corruption --stage prepare
python run_experiment.py controlled-corruption --stage evaluate --device cuda:0
python run_experiment.py controlled-corruption --stage summarize

# Table 6 and deterministic top-K behavior analysis
python run_experiment.py sampler-stability run --device cuda:0
python run_experiment.py sampler-stability summarize
python run_experiment.py rank-movement run --device cuda:0
python run_experiment.py rank-movement summarize

# Late-mask analysis reported with Table 12; run the rank-movement commands first
python run_experiment.py late-mask-analysis collect --device cuda:0
python run_experiment.py late-mask-analysis evaluate --device cuda:0
python run_experiment.py late-mask-analysis summarize

# Exact equivalence at full retention
python run_experiment.py full-retention-check gate --device cuda:0
python run_experiment.py full-retention-check aggregate

# Walmart feature sanity check reported in the experimental appendix
python walmart_feature_check.py run --device cuda:0
python walmart_feature_check.py summarize
```

The saved-mask commands following the core runs produce Tables 10--12.

The Table 5 and Figure 2 runtime study requires the completed core accuracy
matrix and controlled-corruption summary:

```bash
python runtime_study.py --stage run --device cuda:0
python runtime_study.py --stage summarize
```

## Collect Outputs

The main-table script collects Tables 2--4 automatically. After running any
additional experiments, collect the available table inputs, figures, and final
analysis summaries with:

```bash
python collect_outputs.py
```

The command writes the collected files and a checksum manifest under
`./outputs/`. After running the complete experiment suite, add
`--require-complete` to verify that every expected output is present.

## Reproducibility Notes

- Validation accuracy selects hyperparameters and checkpoints. Test results are evaluated only after selection.
- `K = floor(rho * t)` counts selected original incidences. Every sparse method
  receives the same fixed self-loops outside this budget.
- Random-Fixed uses mask seed `100000 + model seed`. Random-Resampled uses a
  separate fixed evaluation mask and a deterministic training-mask stream.
- The runners save masks, per-seed JSON records, aggregate CSV files, and
  hashes of the experiment specifications. They do not require pretrained
  checkpoints.
- GPU kernels and library builds can introduce small floating-point variation;
  use the listed versions and hardware class for the closest reproduction.
