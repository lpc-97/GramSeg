#!/bin/bash
# Batch D — DINOv3-L/16 correlation upgrade, smoke tier (CVPR plan Phase 2a)
# Auto-chains after Batch C: waits for the L weights and batchC/summary.txt.
# Smoke on ADE + VOC21 only; full 8-bench extension is a next-day decision.
source ~/night/common.sh
WL=$HOME/dino_rtp_seg/dinov3_vitl16.safetensors
EXPECT=1212347640
B=$HOME/batchD; mkdir -p "$B"
exec > "$B/run.log" 2>&1
echo "batchD start $(date)"

# -- wait for weights (rename .part when complete) --------------------------
while [ ! -f "$WL" ]; do
  SZ=$(stat -c %s "$WL.part" 2>/dev/null || echo 0)
  if [ "$SZ" = "$EXPECT" ]; then mv "$WL.part" "$WL"; break; fi
  echo "weights $SZ/$EXPECT $(date)"; sleep 60
done
[ "$(stat -c %s "$WL")" = "$EXPECT" ] || { echo "FATAL: weight size mismatch"; exit 1; }
echo "weights ready"

# -- wait for batchC to release GPUs ---------------------------------------
while [ ! -f ~/batchC/summary.txt ]; do sleep 300; done
echo "batchC done, starting $(date)"
GPU=3
touch ~/.gpu_claim_gpu${GPU}_batchD; trap 'rm -f ~/.gpu_claim_gpu'${GPU}'_batchD' EXIT

evL() { # $1 cfg, $2 upscale, rest cfg-options
  local cfg=$1 up=$2; shift 2
  (cd $CC && PYTHONNOUSERSITE=1 DINOV3_WEIGHTS=$WL DINOV3_CORR_UPSCALE=$up \
    CUDA_VISIBLE_DEVICES=$GPU "$PY_EVAL" eval.py --config ./configs/${cfg}.py \
    --cfg-options model.model_type=ViT-L-14-quickgelu model.dino_type=dinov3_vitl16 "$@" 2>&1)
}

MADE=$CC/data/region_masks_gram/ade/$TAG
MVOC=$CC/data/region_masks_gram/voc/$TAG

echo "== 1. L-corr dual, upscale 1.0, ADE (ref: hub DINO-B/8 27.15) =="
evL cfg_ade20k 1.0 model.instance_mask_path=$MADE | tee "$LOG/eval_cfg_ade20k_corrL_u1.log"
echo "== 2. L-corr dual, upscale 2.0, ADE =="
evL cfg_ade20k 2.0 model.instance_mask_path=$MADE | tee "$LOG/eval_cfg_ade20k_corrL_u2.log"
echo "== 3. L-corr dual, best-known upscale on VOC21 (ref 66.59): pick u by ADE =="
A1=$(grep -o 'mIoU: [0-9.]*' "$LOG/eval_cfg_ade20k_corrL_u1.log" | tail -1 | cut -d' ' -f2)
A2=$(grep -o 'mIoU: [0-9.]*' "$LOG/eval_cfg_ade20k_corrL_u2.log" | tail -1 | cut -d' ' -f2)
UBEST=1.0; awk "BEGIN{exit !($A2 > $A1)}" && UBEST=2.0
echo "ADE u1=$A1 u2=$A2 -> UBEST=$UBEST"
evL cfg_voc21 $UBEST model.instance_mask_path=$MVOC | tee "$LOG/eval_cfg_voc21_corrL_u$UBEST.log"

echo "== 4. unified-L: gen ADE masks with vitl16 (shipped refined config) =="
LTAG=vitl16_k64_hard_ms448_t0.6_pamr10r1024
PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=$GPU "$PY_GEN" \
  ~/gram_corrclip/gen_gram_masks.py --images $CC/data/ade/ADEChallengeData2016/images/validation \
  --out $CC/data/region_masks_gram/ade/$LTAG --weights $WL --variant vitl16 \
  --k 64 --binarize hard --img-size 896 --upsample nearest \
  --multiscale --coarse-size 448 --merge-thresh 0.6 --pamr-iters 10 --pamr-res 1024
echo "== 5. unified-L eval ADE (single 300M model row; ref unified-B 26.68) =="
evL cfg_ade20k $UBEST model.instance_mask_path=$CC/data/region_masks_gram/ade/$LTAG \
  | tee "$LOG/eval_cfg_ade20k_fullL_u$UBEST.log"

echo "== BATCHD SUMMARY ==" | tee "$B/summary.txt"
for f in $LOG/eval_cfg_ade20k_corrL_u1.log $LOG/eval_cfg_ade20k_corrL_u2.log \
         $LOG/eval_cfg_voc21_corrL_u*.log $LOG/eval_cfg_ade20k_fullL_u*.log; do
  printf '%-50s %s\n' "$(basename $f)" "$(grep -o 'mIoU: [0-9.]*' $f 2>/dev/null | tail -1)" | tee -a "$B/summary.txt"
done
echo "BATCHD_DONE $(date)" | tee -a "$B/summary.txt"
