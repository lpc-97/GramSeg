#!/bin/bash
# 过夜 GPU0 链:等词表 v3 完 → Batch V 前半(ours-base 在线 d3 全量:VOC21/VOC20/ADE/City/Stuff)
source ~/night/common.sh
mkdir -p ~/batchV
exec > ~/night_gpu0.log 2>&1
echo "night_gpu0 start $(date)"

for i in $(seq 1 240); do grep -q "BATCHU_VOCAB3_DONE" ~/batchU/eff_chain.log && break; sleep 60; done

bv() { name=$1; cfg=$2; cd $CC && PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
  GRAM_MCT_DILATE=3 GRAM_MS=0 GRAM_PAMR=0 \
  $PY_EVAL eval.py --config ./configs/$cfg.py \
  --cfg-options model.mask_generator=gram model.model_type=ViT-L-14-quickgelu \
  > ~/batchV/$name.log 2>&1; \
  echo "$name $(grep -oE 'mIoU: [0-9.]+' ~/batchV/$name.log | tail -1)" >> ~/batchV/summary.txt; }

bv base_voc21 cfg_voc21
bv base_voc20 cfg_voc20
bv base_ade cfg_ade20k
bv base_city cfg_city_scapes
bv base_stuff cfg_coco_stuff164k
echo "NIGHT_GPU0_DONE $(date)"
