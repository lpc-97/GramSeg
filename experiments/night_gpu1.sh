#!/bin/bash
# 过夜 GPU1 链:dinotxt 分辨率扫描(ADE 768/896)→ City 官方校准 → Batch V 后半(ours-base 全量)
source ~/night/common.sh
mkdir -p ~/batchV
exec > ~/night_gpu1.log 2>&1
echo "night_gpu1 start $(date)"

HEAD_SNAP=$(find ${CACHE_ROOT:-$HOME/cache}/dinotxt_cache -name "dinov3_vitl16_dinotxt_vision_head_and_text_encoder.pth" | head -1)
BPE=$(find ${CACHE_ROOT:-$HOME/cache}/dinotxt_cache -name "bpe_simple_vocab_16e6.txt.gz" | head -1)
BBW=~/dino_rtp_seg/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth

# 等 nbnames 校准结束
for i in $(seq 1 120); do grep -q "DINOTXT_NBNAMES_DONE" ~/dinotxt_chain2.log && break; sleep 60; done

dt() { tag=$1; shift; PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=1 ~/miniconda3/envs/bl/bin/python \
  ~/dinotxt_ade_eval.py --head-weights "$HEAD_SNAP" --backbone-weights "$BBW" --bpe "$BPE" "$@" \
  > ~/batchU/dinotxt_$tag.log 2>&1; echo "dinotxt_$tag rc=$? $(grep FINAL ~/batchU/dinotxt_$tag.log)" ; }

dt ade_r768 --mode official --resize 768 --names ~/nb_ade_names.txt
dt ade_r896 --mode official --resize 896 --names ~/nb_ade_names.txt
dt city_r512 --mode official --dataset city --resize 512
echo "DINOTXT_SWEEP_DONE $(date)"

# Batch V 后半:ours-base(在线,d3,GRAM_MS=0 GRAM_PAMR=0)
bv() { name=$1; cfg=$2; cd $CC && PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 \
  GRAM_MCT_DILATE=3 GRAM_MS=0 GRAM_PAMR=0 \
  $PY_EVAL eval.py --config ./configs/$cfg.py \
  --cfg-options model.mask_generator=gram model.model_type=ViT-L-14-quickgelu \
  > ~/batchV/$name.log 2>&1; \
  echo "$name $(grep -oE 'mIoU: [0-9.]+' ~/batchV/$name.log | tail -1)" >> ~/batchV/summary.txt; }

bv base_object cfg_coco_object
bv base_pc59 cfg_context59
bv base_pc60 cfg_context60
echo "NIGHT_GPU1_DONE $(date)"
