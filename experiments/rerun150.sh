#!/bin/bash
source ~/night/common.sh
B=$HOME/batchU
run_timed() {
  tag=$1; shift
  echo "== EFF $tag start $(date +%s) $(date) ==" >> "$B/eff_chain.log"
  ( while true; do nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits; sleep 1; done > "$B/mem_$tag.log" ) & POLL=$!
  t0=$(date +%s.%N)
  "$@" > "$B/eff_$tag.log" 2>&1
  rc=$?
  t1=$(date +%s.%N)
  kill $POLL 2>/dev/null; wait $POLL 2>/dev/null
  peak=$(sort -n "$B/mem_$tag.log" | tail -1)
  echo "== EFF $tag done rc=$rc wall=$(awk "BEGIN{print $t1 - $t0}")s peakmem=${peak}MiB ==" >> "$B/eff_chain.log"
}
ours_v() { cd $CC && PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
  GRAM_MCT_DILATE=3 GRAM_VOCAB_MEMFIX=1 \
  $PY_EVAL eval.py --config ./configs/cfg_ade20k_500.py \
  --cfg-options model.mask_generator=gram model.model_type=ViT-L-14-quickgelu \
  model.name_path=./configs/cls_ade20k.txt; }
run_timed ours_vl150_memfix_v2 ours_v
echo "RERUN150_DONE $(date)" >> "$B/eff_chain.log"
