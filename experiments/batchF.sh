#!/bin/bash
# Batch F — SOTA-number batch (2026-07-15 午后):
#  chain_mb16 (GPU0): MetaCLIP-B16 remaining 7 cols (ADE 23.52 done) -> "ours best B/16" row
#  chain_h   (GPU3): download MetaCLIP H14 (official, 11.8GB) -> ADE + VOC21 smoke ("ours-H")
source ~/night/common.sh
B=$HOME/batchF; mkdir -p "$B"
H14=$HOME/.cache/clip/h14_fullcc2.5b.pt
H14_SIZE=11834126175

chain_mb16() {
  exec > "$B/mb16.log" 2>&1
  touch ~/.gpu_claim_gpu0_batchF; trap 'rm -f ~/.gpu_claim_gpu0_batchF' EXIT
  echo "chain_mb16 start $(date)"
  declare -A DSOF=( [cfg_voc21]=voc [cfg_voc20]=voc [cfg_coco_stuff164k]=coco \
                    [cfg_coco_object]=coco [cfg_context59]=pc [cfg_context60]=pc \
                    [cfg_city_scapes]=city )
  for cfg in cfg_voc21 cfg_voc20 cfg_coco_stuff164k cfg_coco_object cfg_context59 cfg_context60 cfg_city_scapes; do
    D=${DSOF[$cfg]}
    echo "== MetaCLIP-B16 $cfg =="
    ev $cfg 0 model.model_type=ViT-B-16-quickgelu \
      model.instance_mask_path=$CC/data/region_masks_gram/$D/$TAG \
      | tee "$LOG/eval_${cfg}_metab16.log"
  done
  echo "MB16_DONE $(date)"
}

chain_h() {
  exec > "$B/h14.log" 2>&1
  touch ~/.gpu_claim_gpu3_batchF; trap 'rm -f ~/.gpu_claim_gpu3_batchF' EXIT
  echo "chain_h start $(date)"
  if [ ! -f "$H14" ] || [ "$(stat -c %s "$H14")" != "$H14_SIZE" ]; then
    echo "downloading H14 (11.8GB, official dl.fbaipublicfiles)"
    curl -sL --retry 5 -C - -o "$H14.part" \
      "https://dl.fbaipublicfiles.com/MMPT/metaclip/h14_fullcc2.5b.pt"
    SZ=$(stat -c %s "$H14.part")
    [ "$SZ" = "$H14_SIZE" ] || { echo "FATAL: size $SZ != $H14_SIZE"; exit 1; }
    mv "$H14.part" "$H14"
    sha256sum "$H14" | tee "$B/h14.sha256"   # provenance record (official direct)
  fi
  echo "H14 ready $(date)"
  echo "== MetaCLIP-H14 ADE (refs: ours-L 27.15, ProxyCLIP-H avg 44.4) =="
  ev cfg_ade20k 3 model.model_type=ViT-H-14-quickgelu \
    model.instance_mask_path=$CC/data/region_masks_gram/ade/$TAG \
    | tee "$LOG/eval_cfg_ade20k_metah14.log" || echo H14_ADE_FAILED
  echo "== MetaCLIP-H14 VOC21 (ref ours-L 66.59) =="
  ev cfg_voc21 3 model.model_type=ViT-H-14-quickgelu \
    model.instance_mask_path=$CC/data/region_masks_gram/voc/$TAG \
    | tee "$LOG/eval_cfg_voc21_metah14.log" || echo H14_VOC_FAILED
  echo "H_DONE $(date)"
}

chain_mb16 & chain_h &
wait
echo "== BATCHF SUMMARY $(date) ==" | tee "$B/summary.txt"
for f in eval_cfg_voc21_metab16 eval_cfg_voc20_metab16 eval_cfg_coco_stuff164k_metab16 \
         eval_cfg_coco_object_metab16 eval_cfg_context59_metab16 eval_cfg_context60_metab16 \
         eval_cfg_city_scapes_metab16 eval_cfg_ade20k_metah14 eval_cfg_voc21_metah14; do
  printf '%-40s %s\n' "$f" "$(grep -o 'mIoU: [0-9.]*' $LOG/$f.log 2>/dev/null | tail -1)" | tee -a "$B/summary.txt"
done
echo "BATCHF_ALL_DONE $(date)" | tee -a "$B/summary.txt"
