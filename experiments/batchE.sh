#!/bin/bash
# Batch E — retries with fixes (2026-07-15):
#  chain_L (GPU3): DINOv3-L correlation (bf16 fix) + L masks; smoke NaN check first
#  chain_retry (GPU1): topp-base ADE + MetaCLIP-B16 ADE (both were OOM victims)
source ~/night/common.sh
WL=$HOME/dino_rtp_seg/dinov3_vitl16.safetensors
B=$HOME/batchE; mkdir -p "$B"

evL() { # $1 cfg, $2 gpu, $3 upscale, rest cfg-options
  local cfg=$1 gpu=$2 up=$3; shift 3
  (cd $CC && PYTHONNOUSERSITE=1 DINOV3_WEIGHTS=$WL DINOV3_CORR_UPSCALE=$up \
    CUDA_VISIBLE_DEVICES=$gpu "$PY_EVAL" eval.py --config ./configs/${cfg}.py \
    --cfg-options model.model_type=ViT-L-14-quickgelu model.dino_type=dinov3_vitl16 "$@" 2>&1)
}

chain_L() {
  exec > "$B/L.log" 2>&1
  touch ~/.gpu_claim_gpu3_batchE; trap 'rm -f ~/.gpu_claim_gpu3_batchE' EXIT
  echo "chain_L start $(date)"

  echo "== 0. bf16 NaN smoke (30s) =="
  PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=3 "$PY_EVAL" - <<'PYEOF'
import torch, timm, os
m = timm.create_model('vit_large_patch16_dinov3.lvd1689m', pretrained=True,
    num_classes=0, dynamic_img_size=True,
    pretrained_cfg_overlay=dict(file=os.path.expanduser('~/dino_rtp_seg/dinov3_vitl16.safetensors')))
m = m.eval().cuda().to(torch.bfloat16)
x = torch.randn(1, 3, 448, 448, device='cuda', dtype=torch.bfloat16)
with torch.no_grad():
    f16 = m.to(torch.float16).forward_features(x.half())
    fbf = m.to(torch.bfloat16).forward_features(x.to(torch.bfloat16))
print('fp16 finite:', torch.isfinite(f16).all().item(),
      '| bf16 finite:', torch.isfinite(fbf).all().item())
assert torch.isfinite(fbf).all(), 'bf16 still NaN -- abort'
print('SMOKE_OK')
PYEOF
  [ $? -eq 0 ] || { echo "SMOKE_FAILED, aborting chain_L"; exit 1; }

  echo "== 1. L-corr dual u1.0 ADE (ref hub-B/8: 27.15) =="
  evL cfg_ade20k 3 1.0 model.instance_mask_path=$CC/data/region_masks_gram/ade/$TAG \
    | tee "$LOG/eval_cfg_ade20k_corrL_u1.log"
  echo "== 2. L-corr dual u2.0 ADE =="
  evL cfg_ade20k 3 2.0 model.instance_mask_path=$CC/data/region_masks_gram/ade/$TAG \
    | tee "$LOG/eval_cfg_ade20k_corrL_u2.log"
  A1=$(grep -o 'mIoU: [0-9.]*' "$LOG/eval_cfg_ade20k_corrL_u1.log" | tail -1 | cut -d' ' -f2)
  A2=$(grep -o 'mIoU: [0-9.]*' "$LOG/eval_cfg_ade20k_corrL_u2.log" | tail -1 | cut -d' ' -f2)
  UBEST=1.0; awk "BEGIN{exit !(${A2:-0} > ${A1:-0})}" && UBEST=2.0
  echo "ADE u1=$A1 u2=$A2 -> UBEST=$UBEST"
  echo "== 3. L-corr dual VOC21 (ref 66.59) =="
  evL cfg_voc21 3 $UBEST model.instance_mask_path=$CC/data/region_masks_gram/voc/$TAG \
    | tee "$LOG/eval_cfg_voc21_corrL.log"

  LTAG=vitl16_k64_hard_ms448_t0.6_pamr10r1024
  echo "== 4. gen L masks ADE (bf16) =="
  PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=3 "$PY_GEN" \
    ~/gram_corrclip/gen_gram_masks.py --images $CC/data/ade/ADEChallengeData2016/images/validation \
    --out $CC/data/region_masks_gram/ade/$LTAG --weights $WL --variant vitl16 \
    --k 64 --binarize hard --img-size 896 --upsample nearest \
    --multiscale --coarse-size 448 --merge-thresh 0.6 --pamr-iters 10 --pamr-res 1024
  echo "== 5. unified-L eval ADE (ref unified-B 26.68) =="
  evL cfg_ade20k 3 $UBEST model.instance_mask_path=$CC/data/region_masks_gram/ade/$LTAG \
    | tee "$LOG/eval_cfg_ade20k_fullL.log"
  echo "L_DONE $(date)"
}

chain_retry() {
  exec > "$B/retry.log" 2>&1
  touch ~/.gpu_claim_gpu1_batchE; trap 'rm -f ~/.gpu_claim_gpu1_batchE' EXIT
  echo "chain_retry start $(date)"
  echo "== topp base ADE (masks exist; ref hard base 26.49) =="
  ev cfg_ade20k 1 model.model_type=ViT-L-14-quickgelu \
    model.instance_mask_path=$CC/data/region_masks_gram/ade/k64_topp_p0.9_nearest_s896_vits16 \
    | tee "$LOG/eval_cfg_ade20k_topp_base.log"
  echo "== MetaCLIP-B16 ADE (ref openai-B16 20.77) =="
  export HF_ENDPOINT=https://hf-mirror.com HF_HUB_ETAG_TIMEOUT=60
  ev cfg_ade20k 1 model.model_type=ViT-B-16-quickgelu \
    model.instance_mask_path=$CC/data/region_masks_gram/ade/$TAG \
    | tee "$LOG/eval_cfg_ade20k_metab16.log" || echo METAB16_FAILED
  echo "RETRY_DONE $(date)"
}

chain_L & chain_retry &
wait
echo "== BATCHE SUMMARY $(date) ==" | tee "$B/summary.txt"
for f in eval_cfg_ade20k_corrL_u1 eval_cfg_ade20k_corrL_u2 eval_cfg_voc21_corrL \
         eval_cfg_ade20k_fullL eval_cfg_ade20k_topp_base eval_cfg_ade20k_metab16; do
  printf '%-40s %s\n' "$f" "$(grep -o 'mIoU: [0-9.]*' $LOG/$f.log 2>/dev/null | tail -1)" | tee -a "$B/summary.txt"
done
echo "BATCHE_ALL_DONE $(date)" | tee -a "$B/summary.txt"
