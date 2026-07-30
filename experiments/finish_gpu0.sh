#!/bin/bash
# GPU0 收尾链:SAM2 探针(autocast 修复)→ memfix@150 对照 → ours 5k → scclip 5k
source ~/night/common.sh
B=$HOME/batchU
exec > ~/finish_gpu0.log 2>&1
echo "finish_gpu0 start $(date)"
PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
  ~/miniconda3/envs/corrclip/bin/python ~/probe_sam2_maskside.py > ~/batchU/sam2_probe.log 2>&1
echo "SAM2_PROBE_V2_DONE rc=$? $(grep FINAL ~/batchU/sam2_probe.log)"
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
  model.name_path=./configs/$1; }
scv() { cd ~/SC-CLIP && PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
  ~/miniconda3/envs/corrclip/bin/python eval.py --config ./configs/$1 --work-dir $B/eff_scv5_wd; }
run_timed ours_vl150_memfix ours_v cls_ade20k.txt
run_timed ours_vl5k_memfix ours_v cls_vlines5k.txt
run_timed scclip_vl5k scv cfg_ade20k_l_500_vl5k.py
echo "FINISH_GPU0_DONE $(date)"
