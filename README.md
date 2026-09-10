# EdgeMask-HGNN

Source codes for the paper titled: **Supervised Incidence Sparsification of
Hypergraphs under Budget Constraints**.

Here we provide the experiment
specifications, runners, and aggregation code used for the reported results. Note that, it is recommended to run all commands from the repository root.

## Setup

Tested with Python 3.9, PyTorch 2.0.0, CUDA 11.8, and 32 GB NVIDIA V100 GPU.

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

## Datasets

`data/data.zip` contains the node-classification datasets. Datasets for
Hyper-SAGNN hyperedge prediction are downloaded from the original authors'
codebase. The following commands prepare the datasets:

```bash
python scripts/extract_node_data.py data/data.zip
python scripts/fetch_hypersagnn_data.py
python check_data.py --scope all
```

The Hyper-SAGNN downloader is pinned to upstream commit
[`69f2fbe`](https://github.com/ma-compbio/Hyper-SAGNN/commit/69f2fbe21c455aca084497fb2d26a8207a95decd)
and verifies every downloaded file. Hyper-SAGNN attributes these datasets to
[DHNE](https://github.com/tadpole/DHNE).

Generate the dataset statistics in Table 1 with:

```bash
python dataset_statistics.py
```

## Quick Check

Run one EHGNN-F seed on Actor:

```bash
python run_model.py --model ehgnn-f --dataset actor --budget 0.5 --seed 0
```

Outputs are written under `outputs/single_runs/`. Run
`python run_model.py --help` for the available models, datasets, and budgets.

## Main Results

### Tables 2--4 and mask analyses

```bash
DEVICE=cuda:0 bash scripts/run_core_tables.sh
python summarize_mask_movement.py
python run_experiment.py retained-structure
```

These commands generate the main accuracy comparisons and the saved-mask
analyses used in Tables 10--12.

### Table 5 and Figure 2

Run the core tables and controlled-corruption experiment first, then:

```bash
python runtime_study.py --stage run --device cuda:0
python runtime_study.py --stage summarize
```

### Table 7: HSL and HERALD

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
```

The fetch script pins both upstream repositories and applies the documented HSL
forward-mask correction.

### Tables 13--14: controlled corruption

```bash
python run_experiment.py controlled-corruption --stage prepare
python run_experiment.py controlled-corruption --stage evaluate --device cuda:0
python run_experiment.py controlled-corruption --stage summarize
```

### Tables 15--17

Run the core-table workflow first; Table 15 reuses its reference records.

```bash
# Table 15: conditioned scorers
python conditioned_scorer_study.py evaluate \
  --variants f_cond f_cond_lr --device cuda:0
python conditioned_scorer_study.py summarize \
  --variants f_cond f_cond_lr

# Table 16: AllSetTransformer and ED-HNN
python backbone_study.py --device cuda:0
python backbone_study.py --summarize

# Table 17: Hyper-SAGNN hyperedge prediction
python hyperedge_prediction.py --device cuda:0
python hyperedge_prediction.py --summarize
```

## Remaining Analyses

```bash
# Table 6: sampler stability and deterministic top-K behavior
python run_experiment.py sampler-stability run --device cuda:0
python run_experiment.py sampler-stability summarize
python run_experiment.py rank-movement run --device cuda:0
python run_experiment.py rank-movement summarize

# Table 8: robustness over 15 splits
python run_experiment.py split-robustness audit-splits
python run_experiment.py split-robustness evaluate --device cuda:0
python run_experiment.py split-robustness summarize

# Table 9: connected and isolated test nodes
python run_experiment.py connected-nodes run --device cuda:0
python run_experiment.py connected-nodes summarize

# Selected-gradient and late-mask analyses for Table 12
python run_experiment.py selected-gradients run --device cuda:0
python run_experiment.py selected-gradients aggregate
python run_experiment.py late-mask-analysis collect --device cuda:0
python run_experiment.py late-mask-analysis evaluate --device cuda:0
python run_experiment.py late-mask-analysis summarize

# Appendix checks
python run_experiment.py full-retention-check gate --device cuda:0
python run_experiment.py full-retention-check aggregate
python walmart_feature_check.py run --device cuda:0
python walmart_feature_check.py summarize
```

Use `python run_experiment.py --help` and each script's `--help` option to run
smaller subsets or resume individual phases.

## Collecting the Outputs

```bash
python collect_outputs.py
python collect_outputs.py --require-complete
```

The first command collects available tables, figures, summaries, and their
checksums under `outputs/`. Run `--require-complete` only after the full suite;
it fails if any expected deliverable is missing.

## Reproducibility Details

- Fixed settings are stored under `experiment_specs/`.
- Validation accuracy selects hyperparameters and checkpoints; test results are
  evaluated only after selection.
- For sparse methods, `K = floor(rho * t)` counts selected original incidences.
  Fixed self-loops are outside this budget.
- Random-Fixed uses mask seed `100000 + model seed`. Random-Resampled uses a
  deterministic training-mask stream and a separate fixed evaluation mask.
- GPU kernels can cause small floating-point variation. Use the listed software
  versions and hardware class for the closest reproduction.
