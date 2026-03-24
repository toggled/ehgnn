#!/usr/bin/env bash
set -euo pipefail

# =========================
# Output CSV
# =========================
savef="induc_corrupt_family_grid_trainpercent0.1v3.csv"

# =========================
# Base dataset (the clean one)
# train_controlled.py should load this normally, then corrupt dataset.data if --corrupt
# =========================
BASE_DNAME="cora"   # change to your dataset name in dataset_Hypergraph

# =========================
# Sweeps
# =========================
budgets=(0.5 0.75)

# corruption seeds (used INSIDE corrupt_family)
cseeds=(0)

# 2D grid for (p_edge, q_inc)
# coarse-favored: higher p_edge, low q_inc
# fine-favored:   low p_edge, higher q_inc
p_edges=(0.00 0.15 0.30 0.45 0.60)
q_incs=(0.00 0.10 0.20 0.30 0.40)

# GPUs
GPU_C=0   # coarse
GPU_F=1   # fine

# Common training args
COMMON_ARGS=(
  --method HGNN
  --All_num_layers 1
  --MLP_num_layers 2
  --Classifier_num_layers 1
  --MLP_hidden 512
  --Classifier_hidden 256
  --feature_noise 0.0
  --heads 1
  --wd 0.0
  --epochs 1000
  --patience 200
  --runs 5
  --lr 0.0005
  --seed 1
  --fname "${savef}"
  --theory
  --HCHA_symdegnorm
)

# Limit concurrency if desired
MAXJOBS=${MAXJOBS:-8}

wait_for_slots () {
  while true; do
    local nj
    nj=$(jobs -pr | wc -l | tr -d ' ')
    if [[ "${nj}" -lt "${MAXJOBS}" ]]; then
      break
    fi
    sleep 3
  done
}

run_pair () {
  local dname="$1"
  local k="$2"
  local pe="$3"
  local qi="$4"
  local cs="$5"

  echo "============================================================"
  echo "[RUN] dname=${dname} keep_ratio=${k} p_edge=${pe} q_inc=${qi} cseed=${cs}"
  echo "============================================================"

  # ---- EHGNN-C(cond)  (coarse)
  python train_corrupt.py \
    --dname "${BASE_DNAME}" \
    --mode learnmask+ \
    --keep_ratio "${k}" \
    --cuda "${GPU_C}" \
    --corrupt \
    --p_edge "${pe}" \
    --q_inc "${qi}" \
    --corrupt_seed "${cs}" \
    "${COMMON_ARGS[@]}" &

  # ---- EHGNN-F(cond)  (fine)
  python train_corrupt.py \
    --dname "${BASE_DNAME}" \
    --mode learnmask_cond \
    --keep_ratio "${k}" \
    --cuda "${GPU_F}" \
    --corrupt \
    --p_edge "${pe}" \
    --q_inc "${qi}" \
    --corrupt_seed "${cs}" \
    "${COMMON_ARGS[@]}" &
}

# NOTE:
# - --dname stays as BASE_DNAME so dataset loader finds the real dataset.
# - --log_dname is used ONLY for naming/logging/CSV rows (recommended).
#   If you didn't implement --log_dname, then:
#     (a) remove --log_dname lines, AND
#     (b) have train_controlled.py internally rename to the "corrupt_family_..." prefix.

for cs in "${cseeds[@]}"; do
  for pe in "${p_edges[@]}"; do
    for qi in "${q_incs[@]}"; do

      pe_tag=$(printf "%.2f" "${pe}")
      qi_tag=$(printf "%.2f" "${qi}")
      dname="corrupt_family_${BASE_DNAME}_pe${pe_tag}_qi${qi_tag}_seed${cs}"

      for k in "${budgets[@]}"; do
        wait_for_slots
        run_pair "${dname}" "${k}" "${pe}" "${qi}" "${cs}"
      done

    done
  done
done

wait
echo "[DONE] All runs finished. Results appended to: ${savef}"