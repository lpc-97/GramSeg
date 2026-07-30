#!/bin/bash
# 等 SAM2 探针完 → GPU0 重跑 ours 10k(memfix)+ 150 对照(memfix 口径一致性)
source ~/night/common.sh
B=$HOME/batchU
exec >> "$B/eff_chain.log" 2>&1
for i in $(seq 1 600); do grep -q "SAM2_PROBE_DONE" ~/after_gpu0.log 2>/dev/null && break; sleep 60; done
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
  GRAM_MCT_DILATE=3 GRAM_VOCAB_MEMFIX=1 \
  $PY_EVAL eval.py --config ./configs/cfg_ade20k_500.py \
  --cfg-options model.mask_generator=gram model.model_type=ViT-L-14-quickgelu \
  model.name_path=./configs/$1; }
run_timed ours_vl10k_memfix ours_v cls_vlines10k.txt
run_timed ours_vl1k_memfix ours_v cls_vlines1k.txt
echo "BATCHU_MEMFIX_DONE $(date)"
