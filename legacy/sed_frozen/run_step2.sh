#!/usr/bin/env bash
# Step-2 (go/no-go experiment 1): cached-feature adapter for SED-B.
#
# Deploy: scp this directory to the server as ~/SED/step2/, then run from
# the SED repo root with the conda env active and a free GPU:
#   cd ~/SED && CUDA_VISIBLE_DEVICES=<free> bash step2/run_step2.sh
#
# Pipeline per checkpoint (frozen-20k / official):
#   1. cache clip_vis_dense + GT soft labels on 20k COCO-Stuff train images
#   2. train the ~4M adapter on the cache (wall-clock printed -> cost story)
#   3. standard ADE-150 zero-shot eval with the adapter injected
#
# Reference numbers: frozen-20k baseline 24.69 mIoU, official full-FT 31.8.
# Decision line: adapter(frozen) >= 30 strong go; >= 24.69 with sec~min
# training keeps the cost story alive.
set -euo pipefail

SED_ROOT=${SED_ROOT:-$(pwd)}
STEP2_DIR=$(cd "$(dirname "$0")" && pwd)
export DETECTRON2_DATASETS=${DETECTRON2_DATASETS:-d2_datasets}

CONFIG=${CONFIG:-configs/convnextB_768.yaml}
OFFICIAL_CKPT=${OFFICIAL_CKPT:-$HOME/offline_staging/sed/sed_convnextB.pth}
FROZEN_CKPT=${FROZEN_CKPT:-$SED_ROOT/output/frozenB_20k/model_final.pth}
NUM_IMAGES=${NUM_IMAGES:-20000}
CACHE_ROOT=${CACHE_ROOT:-$SED_ROOT/cache_sed}
OUT_ROOT=${OUT_ROOT:-$SED_ROOT/output/step2}

cd "$SED_ROOT"
mkdir -p "$OUT_ROOT"

# ---------------------------------------------------------------- 1) caches
# (resumable; ~15-18GB fp16 per cache for 20k images)
python "$STEP2_DIR/cache_sed_features.py" --config "$CONFIG" \
  --ckpt "$FROZEN_CKPT" --num-images "$NUM_IMAGES" --out "$CACHE_ROOT/frozen" \
  2>&1 | tee -a "$OUT_ROOT/cache_frozen.log"
python "$STEP2_DIR/cache_sed_features.py" --config "$CONFIG" \
  --ckpt "$OFFICIAL_CKPT" --num-images "$NUM_IMAGES" --out "$CACHE_ROOT/official" \
  2>&1 | tee -a "$OUT_ROOT/cache_official.log"

# --------------------------------------------------------------- 2) adapters
python "$STEP2_DIR/train_sed_adapter.py" --cache "$CACHE_ROOT/frozen" \
  2>&1 | tee "$OUT_ROOT/train_adapter_frozen.log"
python "$STEP2_DIR/train_sed_adapter.py" --cache "$CACHE_ROOT/official" \
  2>&1 | tee "$OUT_ROOT/train_adapter_official.log"

# ------------------------------------------------------- 3) ADE-150 evals
# Standard SED protocol (eval.sh ADE20k-150 block), single GPU.
eval_ade150 () {  # $1 weights  $2 outdir  $3 adapter path ("" = baseline)
  local weights=$1 outdir=$2 adapter=${3:-}
  echo "=== eval ADE-150: weights=$weights adapter=${adapter:-<none>} -> $outdir"
  SED_ADAPTER_PATH="$adapter" python "$STEP2_DIR/train_net_adapter.py" \
    --config "$CONFIG" --num-gpus 1 --eval-only \
    OUTPUT_DIR "$outdir" \
    MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "datasets/ade150.json" \
    DATASETS.TEST '("ade20k_150_test_sem_seg",)' \
    MODEL.SEM_SEG_HEAD.POOLING_SIZES "[1,1]" \
    MODEL.WEIGHTS "$weights"
}

# main go/no-go: frozen-20k head + adapter (vs 24.69 baseline)
eval_ade150 "$FROZEN_CKPT"   "$OUT_ROOT/frozen_adapter"   "$CACHE_ROOT/frozen/adapter.pth"
# headroom probe: official full-FT + adapter (vs 31.8 baseline)
eval_ade150 "$OFFICIAL_CKPT" "$OUT_ROOT/official_adapter" "$CACHE_ROOT/official/adapter.pth"

# baseline reproduction (adapter off; expect ~24.69 / ~31.8). Set
# RUN_BASELINES=1 the first time to confirm the injected path is bit-exact.
if [ "${RUN_BASELINES:-0}" = "1" ]; then
  eval_ade150 "$FROZEN_CKPT"   "$OUT_ROOT/frozen_baseline"   ""
  eval_ade150 "$OFFICIAL_CKPT" "$OUT_ROOT/official_baseline" ""
fi

echo "================ summary ================"
grep -H "train_wall_clock" "$OUT_ROOT"/train_adapter_*.log || true
grep -H copypaste "$OUT_ROOT"/*/log.txt || true
