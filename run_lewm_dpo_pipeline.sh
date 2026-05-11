#!/usr/bin/env bash
set -euo pipefail


# ============================================================
# LeWM + DPO Shooting Full Pipeline
# ============================================================
# Usage:
#   bash run_lewm_dpo_pipeline.sh cube
#   bash run_lewm_dpo_pipeline.sh pusht
#   bash run_lewm_dpo_pipeline.sh tworoom
#   bash run_lewm_dpo_pipeline.sh reacher
#
# Optional:
#   bash run_lewm_dpo_pipeline.sh reacher 5000
#   bash run_lewm_dpo_pipeline.sh reacher 20000 32
#   bash run_lewm_dpo_pipeline.sh cube 20000
# Args:
#   $1 dataset name: cube | pusht | tworoom | reacher
#   $2 num_pairs, default=20000
#   $3 num_candidates, default=32
# ============================================================

DATASET="${1:-}"
NUM_PAIRS="${2:-20000}"
NUM_CANDIDATES="${3:-32}"

if [[ -z "${DATASET}" ]]; then
  echo "Usage: bash $0 <cube|pusht|tworoom|reacher> [num_pairs] [num_candidates]"
  exit 1
fi

export STABLEWM_HOME="${STABLEWM_HOME:-./datasets}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

mkdir -p logs

# -----------------------------
# Dataset-specific config
# -----------------------------
case "${DATASET}" in
  cube)
    EVAL_CONFIG="cube"
    TRAIN_DATA="ogb"
    MODEL_DIR="cube"
    POLICY_ARG="cube"
    CTX_LEN=1
    PLAN_HORIZON=5
    ;;
  pusht)
    EVAL_CONFIG="pusht"
    TRAIN_DATA="pusht"
    MODEL_DIR="pusht"
    POLICY_ARG="pusht/lewm"
    CTX_LEN=1
    PLAN_HORIZON=5
    ;;
  tworoom)
    EVAL_CONFIG="tworoom"
    TRAIN_DATA="tworoom"
    MODEL_DIR="tworoom"
    POLICY_ARG="tworoom/lewm"
    CTX_LEN=1
    PLAN_HORIZON=5
    ;;
  reacher)
    EVAL_CONFIG="reacher"
    TRAIN_DATA="dmc"
    MODEL_DIR="reacher"
    POLICY_ARG="reacher/lewm"
    CTX_LEN=1
    PLAN_HORIZON=5
    ;;
  *)
    echo "Unknown dataset: ${DATASET}"
    echo "Available: cube | pusht | tworoom | reacher"
    exit 1
    ;;
esac

PAIR_NAME="${DATASET}_pairs_eval_dataset_${NUM_PAIRS}"
if [[ "${NUM_CANDIDATES}" != "32" ]]; then
  PAIR_NAME="${PAIR_NAME}_c${NUM_CANDIDATES}"
fi

PAIR_PATH="dpo/${PAIR_NAME}.pt"
DPO_SUBDIR="dpo_plan_${DATASET}_eval_h5_dataset_${NUM_PAIRS}"
if [[ "${NUM_CANDIDATES}" != "32" ]]; then
  DPO_SUBDIR="${DPO_SUBDIR}_c${NUM_CANDIDATES}"
fi
DPO_SUBDIR="${DPO_SUBDIR}_bc005"

echo "============================================================"
echo "Dataset        : ${DATASET}"
echo "Eval config    : ${EVAL_CONFIG}"
echo "Train data     : ${TRAIN_DATA}"
echo "Model dir      : ${MODEL_DIR}"
echo "Policy arg     : ${POLICY_ARG}"
echo "Pairs          : ${PAIR_PATH}"
echo "DPO subdir     : ${DPO_SUBDIR}"
echo "Num pairs      : ${NUM_PAIRS}"
echo "Num candidates : ${NUM_CANDIDATES}"
echo "ctx_len        : ${CTX_LEN}"
echo "plan_horizon   : ${PLAN_HORIZON}"
echo "STABLEWM_HOME  : ${STABLEWM_HOME}"
echo "============================================================"


# ============================================================
# 1. Full CEM baseline
# ============================================================

echo ""
echo "==================== [1] Full CEM baseline ===================="
echo "==================== Skip ===================="
# python eval.py --config-name="${EVAL_CONFIG}" \
#   policy="${POLICY_ARG}" \
#   eval.num_eval=50 \
#   2>&1 | tee "logs/${DATASET}_full_cem_eval50.log"


# ============================================================
# 2. CEM-lite baseline
# ============================================================

echo ""
echo "==================== [2] CEM-lite baseline ===================="
echo "==================== Skip ===================="
# python eval.py --config-name="${EVAL_CONFIG}" \
#   policy="${POLICY_ARG}" \
#   eval.num_eval=50 \
#   solver.num_samples=64 \
#   solver.n_steps=5 \
#   solver.topk=8 \
#   2>&1 | tee "logs/${DATASET}_cem_lite_s64_step5_eval50.log"


# ============================================================
# 3. Generate eval-style DPO pairs
# ============================================================

echo ""
echo "==================== [3] Make DPO pairs ===================="
echo "==================== Skip ===================="

# python make_dpo_pairs_evalstyle.py data="${TRAIN_DATA}" \
#   +dpo.model_dir="${MODEL_DIR}" \
#   +dpo.output="${PAIR_PATH}" \
#   +dpo.num_pairs="${NUM_PAIRS}" \
#   +dpo.num_candidates="${NUM_CANDIDATES}" \
#   +dpo.ctx_len="${CTX_LEN}" \
#   +dpo.plan_horizon="${PLAN_HORIZON}" \
#   +dpo.batch_size=64 \
#   +dpo.pos_strategy=dataset \
#   +dpo.neg_strategy=global_worst \
#   2>&1 | tee "logs/make_${DATASET}_pairs_${NUM_PAIRS}_c${NUM_CANDIDATES}.log"


# ============================================================
# 4. Train DPO planner
# ============================================================

echo ""
echo "==================== [4] Train DPO planner ===================="
echo "==================== Skip ===================="

# python train_dpo_plan.py data="${TRAIN_DATA}" \
#   +dpo_train.pairs="${PAIR_PATH}" \
#   +dpo_train.subdir="${DPO_SUBDIR}" \
#   +dpo_train.max_epochs=20 \
#   +dpo_train.batch_size=256 \
#   +dpo_train.devices=1 \
#   +dpo_train.beta=0.1 \
#   +dpo_train.bc_weight=0.05 \
#   2>&1 | tee "logs/train_dpo_plan_${DATASET}_${NUM_PAIRS}_c${NUM_CANDIDATES}_bc005.log"


# ============================================================
# 5. Offline sanity eval
# ============================================================

echo ""
echo "==================== [5] Offline DPO eval: samples=64 ===================="
echo "==================== Skip ===================="
# python eval_dpo_plan_offline.py data="${TRAIN_DATA}" \
#   +dpo_eval.pairs="${PAIR_PATH}" \
#   +dpo_eval.policy_dir="${DPO_SUBDIR}" \
#   +dpo_eval.model_dir="${MODEL_DIR}" \
#   +dpo_eval.num_samples=64 \
#   2>&1 | tee "logs/offline_${DATASET}_${NUM_PAIRS}_c${NUM_CANDIDATES}_s64.log"


echo ""
echo "==================== [5] Offline DPO eval: samples=128 ===================="
echo "==================== Skip ===================="
# python eval_dpo_plan_offline.py data="${TRAIN_DATA}" \
#   +dpo_eval.pairs="${PAIR_PATH}" \
#   +dpo_eval.policy_dir="${DPO_SUBDIR}" \
#   +dpo_eval.model_dir="${MODEL_DIR}" \
#   +dpo_eval.num_samples=128 \
#   2>&1 | tee "logs/offline_${DATASET}_${NUM_PAIRS}_c${NUM_CANDIDATES}_s128.log"


# # Optional s256 offline eval.
# # Use small batch_size to avoid CUDA attention kernel issues.
# echo ""
# echo "==================== [5] Offline DPO eval: samples=256, batch_size=64 ===================="

# python eval_dpo_plan_offline.py data="${TRAIN_DATA}" \
#   +dpo_eval.pairs="${PAIR_PATH}" \
#   +dpo_eval.policy_dir="${DPO_SUBDIR}" \
#   +dpo_eval.model_dir="${MODEL_DIR}" \
#   +dpo_eval.num_samples=256 \
#   +dpo_eval.batch_size=64 \
#   2>&1 | tee "logs/offline_${DATASET}_${NUM_PAIRS}_c${NUM_CANDIDATES}_s256.log"


# ============================================================
# 6. Real DPO Shooting eval
# ============================================================

echo ""
echo "==================== [6] Real DPO Shooting eval: samples=64 ===================="
echo "==================== Skip ===================="

# python eval_dpo_shooting.py --config-name="${EVAL_CONFIG}" \
#   +dpo_shooting.model_dir="${MODEL_DIR}" \
#   +dpo_shooting.policy_dir="${DPO_SUBDIR}" \
#   +dpo_shooting.num_samples=64 \
#   +dpo_shooting.random_frac=0.0 \
#   +dpo_shooting.batch_size=1 \
#   +dpo_shooting.results_dir="dpo_shooting_${DATASET}_${NUM_PAIRS}_c${NUM_CANDIDATES}_s64_eval50" \
#   eval.num_eval=50 \
#   2>&1 | tee "logs/eval_dpo_shooting_${DATASET}_${NUM_PAIRS}_c${NUM_CANDIDATES}_s64_eval50.log"


echo ""
echo "==================== [6] Real DPO Shooting eval: samples=128 ===================="
echo "==================== Skip ===================="

# python eval_dpo_shooting.py --config-name="${EVAL_CONFIG}" \
#   +dpo_shooting.model_dir="${MODEL_DIR}" \
#   +dpo_shooting.policy_dir="${DPO_SUBDIR}" \
#   +dpo_shooting.num_samples=128 \
#   +dpo_shooting.random_frac=0.0 \
#   +dpo_shooting.batch_size=1 \
#   +dpo_shooting.results_dir="dpo_shooting_${DATASET}_${NUM_PAIRS}_c${NUM_CANDIDATES}_s128_eval50" \
#   eval.num_eval=50 \
#   2>&1 | tee "logs/eval_dpo_shooting_${DATASET}_${NUM_PAIRS}_c${NUM_CANDIDATES}_s128_eval50.log"


echo ""
echo "==================== [6] Real DPO Shooting eval: samples=1024===================="

python eval_dpo_shooting.py --config-name="${EVAL_CONFIG}" \
  +dpo_shooting.model_dir="${MODEL_DIR}" \
  +dpo_shooting.policy_dir="${DPO_SUBDIR}" \
  +dpo_shooting.num_samples=1024 \
  +dpo_shooting.random_frac=0.0 \
  +dpo_shooting.batch_size=1 \
  +dpo_shooting.results_dir="dpo_shooting_${DATASET}_${NUM_PAIRS}_c${NUM_CANDIDATES}_s1024_eval50" \
  eval.num_eval=50 \
  2>&1 | tee "logs/eval_dpo_shooting_${DATASET}_${NUM_PAIRS}_c${NUM_CANDIDATES}_s1024_eval50.log"


echo ""
echo "============================================================"
echo "Pipeline finished for ${DATASET}"
echo "Pairs      : ${STABLEWM_HOME}/${PAIR_PATH}"
echo "Policy     : ${STABLEWM_HOME}/${DPO_SUBDIR}/dpo_policy.pt"
echo "Logs saved : logs/"
echo "============================================================"