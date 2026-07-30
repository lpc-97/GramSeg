#!/bin/bash
# Batch C — CVPR completeness batch (2026-07-15)
#  chain_city (GPU3): Cityscapes 8th column — gen dual+unified masks, 4 evals
#  chain_b16  (GPU0): CLIP ViT-B/16 (openai) on remaining 6 datasets (masks reused)
#  chain_misc (GPU2): openai-L PC59/60 fill + topp(soft)-vs-hard ADE ablation
#                     + optional MetaCLIP-B/16 ADE
source ~/night/common.sh
BTAG=vitb16_k64_hard_ms448_t0.6_pamr10r1024
WB=$HOME/dino_rtp_seg/dinov3_vitb16.safetensors
B=$HOME/batchC; mkdir -p "$B"
CITY_IMGS=$CC/data/city_val_images
CITY_MG=$CC/data/region_masks_gram/city

gen_masks() { # $1 gpu, $2 images, $3 out, $4 weights, $5 variant, extra args...
  local gpu=$1 imgs=$2 out=$3 w=$4 var=$5; shift 5
  PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=$gpu "$PY_GEN" \
    ~/gram_corrclip/gen_gram_masks.py --images "$imgs" --out "$out" \
    --weights "$w" --variant "$var" --k 64 --img-size 896 --upsample nearest \
    --multiscale --coarse-size 448 --merge-thresh 0.6 --pamr-iters 10 --pamr-res 1024 "$@"
}

chain_city() {
  exec > "$B/city.log" 2>&1
  touch ~/.gpu_claim_gpu3_batchC; trap 'rm -f ~/.gpu_claim_gpu3_batchC' EXIT
  echo "chain_city start $(date)"
  mkdir -p "$CITY_IMGS"
  find $CC/data/cityscapes/leftImg8bit/val -name '*.png' -exec ln -sf {} "$CITY_IMGS/" \;
  echo "flat dir: $(ls "$CITY_IMGS" | wc -l) images"

  echo "== gen city dual masks (vits16, shipped refined config) =="
  gen_masks 3 "$CITY_IMGS" "$CITY_MG/$TAG" "$W" vits16 --binarize hard
  echo "== gen city unified masks (vitb16) =="
  gen_masks 3 "$CITY_IMGS" "$CITY_MG/$BTAG" "$WB" vitb16 --binarize hard

  echo "== eval city dual MetaCLIP ViT-L (headline) =="
  ev cfg_city_scapes 3 model.model_type=ViT-L-14-quickgelu \
    model.instance_mask_path=$CITY_MG/$TAG | tee "$LOG/eval_cfg_city_scapes_$TAG.log"
  echo "== eval city unified (all DINOv3-B) =="
  (cd $CC && PYTHONNOUSERSITE=1 DINOV3_WEIGHTS=$WB DINOV3_CORR_UPSCALE=1.0 \
    CUDA_VISIBLE_DEVICES=3 "$PY_EVAL" eval.py --config ./configs/cfg_city_scapes.py \
    --cfg-options model.model_type=ViT-L-14-quickgelu model.dino_type=dinov3_vitb16 \
    model.instance_mask_path=$CITY_MG/$BTAG 2>&1) | tee "$LOG/eval_cfg_city_scapes_fullB.log"
  echo "== eval city dual openai ViT-L (fairness row) =="
  ev cfg_city_scapes 3 model.clip_type=openai model.model_type=ViT-L-14 \
    model.instance_mask_path=$CITY_MG/$TAG | tee "$LOG/eval_cfg_city_scapes_openai.log"
  echo "== eval city dual openai ViT-B/16 =="
  ev cfg_city_scapes 3 model.clip_type=openai model.model_type=ViT-B-16 \
    model.instance_mask_path=$CITY_MG/$TAG | tee "$LOG/eval_cfg_city_scapes_vitb16openai.log"
  echo "CITY_DONE $(date)"
}

chain_b16() {
  exec > "$B/b16.log" 2>&1
  touch ~/.gpu_claim_gpu0_batchC; trap 'rm -f ~/.gpu_claim_gpu0_batchC' EXIT
  echo "chain_b16 start $(date)"
  declare -A DSOF=( [cfg_voc21]=voc [cfg_voc20]=voc [cfg_coco_stuff164k]=coco \
                    [cfg_coco_object]=coco [cfg_context59]=pc [cfg_context60]=pc )
  for cfg in cfg_voc21 cfg_voc20 cfg_coco_stuff164k cfg_coco_object cfg_context59 cfg_context60; do
    D=${DSOF[$cfg]}
    echo "== B/16 openai $cfg =="
    ev $cfg 0 model.clip_type=openai model.model_type=ViT-B-16 \
      model.instance_mask_path=$CC/data/region_masks_gram/$D/$TAG \
      | tee "$LOG/eval_${cfg}_vitb16openai.log"
  done
  echo "B16_DONE $(date)"
}

chain_misc() {
  exec > "$B/misc.log" 2>&1
  touch ~/.gpu_claim_gpu2_batchC; trap 'rm -f ~/.gpu_claim_gpu2_batchC' EXIT
  echo "chain_misc start $(date)"
  echo "== openai-L PC59 =="
  ev cfg_context59 2 model.clip_type=openai model.model_type=ViT-L-14 \
    model.instance_mask_path=$CC/data/region_masks_gram/pc/$TAG | tee "$LOG/eval_cfg_context59_openai.log"
  echo "== openai-L PC60 =="
  ev cfg_context60 2 model.clip_type=openai model.model_type=ViT-L-14 \
    model.instance_mask_path=$CC/data/region_masks_gram/pc/$TAG | tee "$LOG/eval_cfg_context60_openai.log"

  echo "== topp base eval (existing masks, vs hard base 26.49) =="
  ev cfg_ade20k 2 model.model_type=ViT-L-14-quickgelu \
    model.instance_mask_path=$CC/data/region_masks_gram/ade/k64_topp_p0.9_nearest_s896_vits16 \
    | tee "$LOG/eval_cfg_ade20k_topp_base.log"

  TOPPREF=k64_topp_p0.9_nearest_s896_vits16_ms448_t0.6_pamr10r1024
  echo "== gen topp refined ADE masks =="
  gen_masks 2 $CC/data/ade/ADEChallengeData2016/images/validation \
    "$CC/data/region_masks_gram/ade/$TOPPREF" "$W" vits16 --binarize topp --top-p 0.9
  echo "== topp refined eval (vs hard refined 27.15) =="
  ev cfg_ade20k 2 model.model_type=ViT-L-14-quickgelu \
    model.instance_mask_path=$CC/data/region_masks_gram/ade/$TOPPREF \
    | tee "$LOG/eval_cfg_ade20k_topp_refined.log"

  echo "== optional: MetaCLIP ViT-B/16 ADE (may need weight download via mirror) =="
  export HF_ENDPOINT=https://hf-mirror.com HF_HUB_ETAG_TIMEOUT=60
  ev cfg_ade20k 2 model.model_type=ViT-B-16-quickgelu \
    model.instance_mask_path=$CC/data/region_masks_gram/ade/$TAG \
    | tee "$LOG/eval_cfg_ade20k_metab16.log" || echo "METAB16_FAILED (optional, ok)"
  echo "MISC_DONE $(date)"
}

chain_city & chain_b16 & chain_misc &
wait
echo "== BATCHC SUMMARY $(date) ==" | tee "$B/summary.txt"
for f in eval_cfg_city_scapes_$TAG eval_cfg_city_scapes_fullB eval_cfg_city_scapes_openai \
         eval_cfg_city_scapes_vitb16openai eval_cfg_voc21_vitb16openai eval_cfg_voc20_vitb16openai \
         eval_cfg_coco_stuff164k_vitb16openai eval_cfg_coco_object_vitb16openai \
         eval_cfg_context59_vitb16openai eval_cfg_context60_vitb16openai \
         eval_cfg_context59_openai eval_cfg_context60_openai \
         eval_cfg_ade20k_topp_base eval_cfg_ade20k_topp_refined eval_cfg_ade20k_metab16; do
  printf '%-45s %s\n' "$f" "$(grep -o 'mIoU: [0-9.]*' $LOG/$f.log 2>/dev/null | tail -1)" | tee -a "$B/summary.txt"
done
echo "BATCHC_ALL_DONE $(date)" | tee -a "$B/summary.txt"
