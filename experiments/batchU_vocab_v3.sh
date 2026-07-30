#!/bin/bash
source ~/night/common.sh
B=$HOME/batchU
exec >> "$B/eff_chain.log" 2>&1
run_timed() {
  tag=$1; shift
  echo "== EFF $tag start $(date +%s) $(date) =="
  ( while true; do nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits; sleep 1; done > "$B/mem_$tag.log" ) & POLL=$!
  t0=$(date +%s.%N)
  "$@" > "$B/eff_$tag.log" 2>&1
  rc=$?
  t1=$(date +%s.%N)
  kill $POLL 2>/dev/null; wait $POLL 2>/dev/null
  peak=$(sort -n "$B/mem_$tag.log" | tail -1)
  echo "== EFF $tag done rc=$rc wall=$(awk "BEGIN{print $t1 - $t0}")s peakmem=${peak}MiB =="
}
ours_v() { cd $CC && PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
  GRAM_MCT_DILATE=3 \
  $PY_EVAL eval.py --config ./configs/cfg_ade20k_500.py \
  --cfg-options model.mask_generator=gram model.model_type=ViT-L-14-quickgelu \
  model.name_path=./configs/$1; }
scv() { cd ~/SC-CLIP && PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
  ~/miniconda3/envs/corrclip/bin/python eval.py --config ./configs/$1 --work-dir $B/eff_scv3_wd; }
run_timed ours_vl1k ours_v cls_vlines1k.txt
run_timed scclip_vl1k scv cfg_ade20k_l_500_vl1k.py
run_timed ours_vl10k ours_v cls_vlines10k.txt
run_timed scclip_vl10k scv cfg_ade20k_l_500_vl10k.py
echo "BATCHU_VOCAB3_DONE $(date)"
