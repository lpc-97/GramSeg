#!/bin/bash
# 词表 6 格 max_memory_allocated 仪表化重测(GPU0 串行)
source ~/night/common.sh
B=$HOME/batchU
exec > ~/vocab_maxalloc.log 2>&1
ours_m() { tag=$1; vocab=$2; cd $CC && PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
  GRAM_MCT_DILATE=3 GRAM_VOCAB_MEMFIX=1 \
  $PY_EVAL ~/eval_maxalloc_wrapper.py eval.py --config ./configs/cfg_ade20k_500.py \
  --cfg-options model.mask_generator=gram model.model_type=ViT-L-14-quickgelu \
  model.name_path=./configs/$vocab > "$B/ma_$tag.log" 2>&1; \
  echo "$tag rc=$? $(grep MAXALLOC "$B/ma_$tag.log" | tail -1) $(grep -oE "time: [0-9.]+" "$B/ma_$tag.log" | tail -1)"; }
sc_m() { tag=$1; cfg=$2; cd ~/SC-CLIP && PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
  ~/miniconda3/envs/corrclip/bin/python ~/eval_maxalloc_wrapper.py eval.py --config ./configs/$cfg \
  --work-dir $B/ma_wd > "$B/ma_$tag.log" 2>&1; \
  echo "$tag rc=$? $(grep MAXALLOC "$B/ma_$tag.log" | tail -1) $(grep -oE "time: [0-9.]+" "$B/ma_$tag.log" | tail -1)"; }
ours_m ours150 cls_ade20k.txt
ours_m ours1k cls_vlines1k.txt
ours_m ours5k cls_vlines5k.txt
sc_m sc150 cfg_ade20k_l_500.py
sc_m sc1k cfg_ade20k_l_500_vl1k.py
sc_m sc5k cfg_ade20k_l_500_vl5k.py
echo "MAXALLOC_DONE $(date)"
