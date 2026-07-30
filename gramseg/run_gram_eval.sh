#!/usr/bin/env bash
# Go/No-Go experiment #2: CorrCLIP ViT-L + Gram-Seed masks (replacing SAM2).
# Baseline to beat: ViT-L + pre-generated SAM2_32 masks = 30.71 mIoU (ADE-150).
# Decision line: >= 30 mIoU -> GO.
#
# Deploy (local machine):
#   scp -r /mnt/e/seg/scripts/gram_corrclip <server>:~/gram_corrclip
# Run (server):
#   bash ~/gram_corrclip/run_gram_eval.sh
# K sweep:
#   for K in 32 64 128; do K=$K bash ~/gram_corrclip/run_gram_eval.sh; done
# Top-p variant:
#   BINARIZE=topp TOP_P=0.9 bash ~/gram_corrclip/run_gram_eval.sh
#
# Two envs on purpose:
#   * mask generation needs timm >= 1.0.20 (DINOv3) -> the dino_rtp_seg env
#   * eval needs mmseg/mmcv -> corrclip env
# The npz files decouple them (that is why the offline route is low-risk).

set -euo pipefail

# ---------------------------- knobs (env-overridable) ----------------------
GPU=${GPU:-3}
K=${K:-64}                        # FPS seeds: sweep 32 / 64 / 128
BINARIZE=${BINARIZE:-hard}        # hard | topp
TOP_P=${TOP_P:-0.9}
CONF=${CONF:-0.0}                 # >0: low-confidence tokens -> 0 (unsegmented)
IMG_SIZE=${IMG_SIZE:-896}         # DINOv3 input longest side (snapped to /16)
UPSAMPLE=${UPSAMPLE:-nearest}     # nearest | soft
VARIANT=${VARIANT:-vits16}        # vits16 | vitb16

# TODO 上服务器后先 `ls ~/dino_rtp_seg/` 确认 safetensors 实际文件名再跑
WEIGHTS=${WEIGHTS:-$HOME/dino_rtp_seg/dinov3_vits16.safetensors}

# python of the env that runs dino_rtp_seg proto (timm with DINOv3 support);
# 上服务器后用 `conda env list` 确认名字(记忆里是 bl),必要时 GEN_PY= 覆盖
GEN_PY=${GEN_PY:-$HOME/miniconda3/envs/bl/bin/python}
EVAL_PY=${EVAL_PY:-$HOME/miniconda3/envs/corrclip/bin/python}

# ---------------------------- derived paths --------------------------------
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
CORRCLIP=$HOME/CorrCLIP
ADE_IMG=$CORRCLIP/data/ade/ADEChallengeData2016/images/validation
TAG=k${K}_${BINARIZE}
if [ "$BINARIZE" = topp ]; then TAG=${TAG}_p${TOP_P}; fi
if [ "$CONF" != "0.0" ] && [ "$CONF" != "0" ]; then TAG=${TAG}_c${CONF}; fi
TAG=${TAG}_${UPSAMPLE}_s${IMG_SIZE}_${VARIANT}
MASK_DIR=$CORRCLIP/data/region_masks_gram/ade/$TAG
LOG=$SCRIPT_DIR/eval_gram_${TAG}.log

echo "=== Gram-Seed x CorrCLIP | TAG=$TAG | GPU=$GPU ==="
[ -f "$WEIGHTS" ] || { echo "FATAL: weights not found: $WEIGHTS (ls ~/dino_rtp_seg/)"; exit 1; }
[ -d "$ADE_IMG" ] || { echo "FATAL: ADE val images not found: $ADE_IMG"; exit 1; }
[ -x "$GEN_PY" ] || { echo "FATAL: GEN_PY not found: $GEN_PY (conda env list)"; exit 1; }

# ---------------------------- step 1: generate masks -----------------------
N_IMG=$(ls "$ADE_IMG" | grep -ci '\.jpg$' || true)
N_NPZ=$(ls "$MASK_DIR" 2>/dev/null | grep -c '\.npz$' || true)
if [ "$N_NPZ" -ge "$N_IMG" ] && [ "$N_IMG" -gt 0 ]; then
    echo "[1/3] masks already complete ($N_NPZ/$N_IMG), skipping generation"
else
    echo "[1/3] generating Gram-Seed masks -> $MASK_DIR ($N_NPZ/$N_IMG done)"
    PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=$GPU "$GEN_PY" \
        "$SCRIPT_DIR/gen_gram_masks.py" \
        --images "$ADE_IMG" --out "$MASK_DIR" --weights "$WEIGHTS" \
        --variant "$VARIANT" --k "$K" --binarize "$BINARIZE" --top-p "$TOP_P" \
        --conf-thresh "$CONF" --img-size "$IMG_SIZE" --upsample "$UPSAMPLE"
    N_NPZ=$(ls "$MASK_DIR" | grep -c '\.npz$')
    [ "$N_NPZ" -ge "$N_IMG" ] || { echo "FATAL: only $N_NPZ/$N_IMG npz generated"; exit 1; }
fi

# ---------------------------- step 2: eval ---------------------------------
# Same command shape as the reproduced 30.71 baseline; only
# instance_mask_path changes (mask_generator stays None in the config).
echo "[2/3] CorrCLIP eval (ViT-L, ADE-150) -> $LOG"
cd "$CORRCLIP"
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=$GPU "$EVAL_PY" eval.py \
    --config ./configs/cfg_ade20k.py \
    --cfg-options model.model_type=ViT-L-14-quickgelu \
                  model.instance_mask_path="$MASK_DIR" \
    2>&1 | tee "$LOG"

# ---------------------------- step 3: verdict ------------------------------
echo "[3/3] result"
MIOU=$(grep -oE 'mIoU: *[0-9]+\.[0-9]+' "$LOG" | tail -1 | grep -oE '[0-9]+\.[0-9]+' || true)
if [ -z "$MIOU" ]; then
    echo "could not parse mIoU from $LOG -- check the log"; exit 1
fi
echo "TAG=$TAG  mIoU=$MIOU  (SAM2_32 baseline: 30.71, go line: 30.0)"
awk -v m="$MIOU" 'BEGIN { exit !(m >= 30.0) }' \
    && echo ">>> GO: Gram-Seed masks clear the 30.0 line" \
    || echo ">>> NO-GO at this setting: below 30.0 (try other K / binarize / img-size)"
