#!/usr/bin/env bash
# Step-2 distillation variant: feature distillation adapter + weight surgery.
export PYTHONPATH=$HOME/SED
#
# Follow-up to run_step2.sh (CE proxy loss: token-acc +9.6pp but mIoU drops,
# 31.74->24.17 / 24.69->18.37 — decoder co-adaptation breaks the cost volume).
# Here the adapter is trained as pure distillation f: frozen(LAION) feature ->
# official fine-tuned feature, and evaluated on a surgery ckpt (official
# decoder + LAION CLIP visual), so the adapter input at eval matches its
# training input exactly.
#
# Requires both paired caches already built (same 20k images, same shards):
#   $CACHE_ROOT/frozen/   from output/frozenB_20k/model_final.pth (LAION visual)
#   $CACHE_ROOT/official/ from ~/offline_staging/sed/sed_convnextB.pth
#
# Deploy next to the other step2 files as ~/SED/step2/, then:
#   cd ~/SED && CUDA_VISIBLE_DEVICES=<free> bash step2/run_step2_distill.sh
#
# Three evals, all standard ADE-150 protocol:
#   (a) surgery ckpt, no adapter    — official decoder + LAION CLIP; the gap
#       to 31.74 quantifies what CLIP fine-tuning contributes. Note the
#       adapter only sits on the cost-volume path (clip_vis_dense before
#       correlation); the guidance/fusion features stay LAION in (a) and (b).
#   (b) surgery ckpt + distill adapter — core experiment: how much of that
#       gap the ~70s adapter buys back. (b) >= 30 -> strong evidence;
#       (b)-(a) = fine-tuning benefit recovered by the adapter.
#   (c) official ckpt + distill adapter — sanity: adapter should be
#       near-identity on already-fine-tuned features, expect ~31.74.
#       Set RUN_SANITY=0 to skip.
set -euo pipefail

SED_ROOT=${SED_ROOT:-$(pwd)}
STEP2_DIR=$(cd "$(dirname "$0")" && pwd)
export DETECTRON2_DATASETS=${DETECTRON2_DATASETS:-d2_datasets}
PY=${PY:-$HOME/miniconda3/envs/bl/bin/python}

CONFIG=${CONFIG:-configs/convnextB_768.yaml}
OFFICIAL_CKPT=${OFFICIAL_CKPT:-$HOME/offline_staging/sed/sed_convnextB.pth}
FROZEN_CKPT=${FROZEN_CKPT:-$SED_ROOT/output/frozenB_20k/model_final.pth}
CACHE_ROOT=${CACHE_ROOT:-$SED_ROOT/cache_sed}
OUT_ROOT=${OUT_ROOT:-$SED_ROOT/output/step2}
ADAPTER=${ADAPTER:-$OUT_ROOT/adapter_distill.pth}
SURGERY=${SURGERY:-$OUT_ROOT/surgery_laionclip_officialrest.pth}
RUN_SANITY=${RUN_SANITY:-1}

cd "$SED_ROOT"
mkdir -p "$OUT_ROOT"

# --------------------------------------------- 1) distillation training (~70s
# optimization; loads both 20k caches, ~30GB RAM; wall-clock printed)
if [ ! -f "$ADAPTER" ]; then
  "$PY" "$STEP2_DIR/train_sed_adapter_distill.py" \
    --frozen "$CACHE_ROOT/frozen" --official "$CACHE_ROOT/official" \
    --out "$ADAPTER" \
    2>&1 | tee "$OUT_ROOT/train_adapter_distill.log"
else
  echo "[skip] $ADAPTER exists (delete to retrain)"
fi

# ------------------------------------------------------- 2) weight surgery
# official ckpt with clip_model.visual.* swapped back to the LAION originals
# (taken from the frozen-training ckpt, whose visual was never updated)
if [ ! -f "$SURGERY" ]; then
  "$PY" "$STEP2_DIR/make_surgery_ckpt.py" \
    --official "$OFFICIAL_CKPT" --frozen "$FROZEN_CKPT" --out "$SURGERY" \
    2>&1 | tee "$OUT_ROOT/make_surgery.log"
else
  echo "[skip] $SURGERY exists (delete to redo surgery)"
fi

# ------------------------------------------------------- 3) ADE-150 evals
# Standard SED protocol (same as run_step2.sh), single GPU.
eval_ade150 () {  # $1 weights  $2 outdir  $3 adapter path ("" = no adapter)
  local weights=$1 outdir=$2 adapter=${3:-}
  echo "=== eval ADE-150: weights=$weights adapter=${adapter:-<none>} -> $outdir"
  SED_ADAPTER_PATH="$adapter" "$PY" "$STEP2_DIR/train_net_adapter.py" \
    --config "$CONFIG" --num-gpus 1 --eval-only \
    OUTPUT_DIR "$outdir" \
    MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "datasets/ade150.json" \
    DATASETS.TEST '("ade20k_150_test_sem_seg",)' \
    MODEL.SEM_SEG_HEAD.POOLING_SIZES "[1,1]" \
    MODEL.WEIGHTS "$weights"
}

# (a) fine-tuned-CLIP contribution: official rest + LAION visual, no adapter
eval_ade150 "$SURGERY" "$OUT_ROOT/surgery_noadapter" ""
# (b) core: can the distilled adapter buy the fine-tuning back?
eval_ade150 "$SURGERY" "$OUT_ROOT/surgery_distill" "$ADAPTER"
# (c) sanity: adapter ~identity on official features (expect ~31.74)
if [ "$RUN_SANITY" = "1" ]; then
  eval_ade150 "$OFFICIAL_CKPT" "$OUT_ROOT/official_distill" "$ADAPTER"
fi

echo "================ summary ================"
grep -H "train_wall_clock" "$OUT_ROOT/train_adapter_distill.log" || true
grep -H "val AFTER" "$OUT_ROOT/train_adapter_distill.log" || true
grep -H "SURGERY_DONE\|replaced" "$OUT_ROOT/make_surgery.log" || true
for d in surgery_noadapter surgery_distill official_distill; do
  [ -f "$OUT_ROOT/$d/log.txt" ] && grep -H copypaste "$OUT_ROOT/$d/log.txt" | tail -1
done
echo "decision: (b) surgery_distill >= 30 mIoU -> strong evidence;"
echo "          (b)-(a) = fine-tuning benefit recovered by the ~70s adapter"
