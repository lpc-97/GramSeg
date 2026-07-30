#!/bin/bash
# batchU 补跑:GPU0 串行 corrclip_sam2(iopath 修复)+ cliper(setuptools 修复),同 run_timed 口径
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

corrclip() { cd $CC && PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
  $PY_EVAL eval.py --config ./configs/cfg_ade20k_500.py \
  --cfg-options model.mask_generator=sam2 model.model_type=ViT-L-14-quickgelu; }

cliper() { cd ~/cliper.code/ovs && PYTHONNOUSERSITE=1 PYTHONPATH=$HOME/cliper.code \
  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 ~/miniconda3/envs/bl/bin/python main_ovs.py \
  --cfg-path ../scripts/config/vit-l-14/ovs_ade150_500.yaml --log-path $B/cliper_log; }

run_timed corrclip_sam2_v2 corrclip
run_timed cliper_v2 cliper
echo "BATCHU_RERUN2_DONE $(date)"
