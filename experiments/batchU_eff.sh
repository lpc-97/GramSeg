#!/bin/bash
# Batch U — Phase 3b 四系统同卡端到端效率表(GPU0 独占串行,ade500)
# 口径:全管线(掩码生成在线 + 分类 + 精修),batch=1,同一 3090(GPU0)。
# 每系统记录:总 wall-clock、mmseg/自报 per-iter time、峰值显存(nvidia-smi 1s 轮询 GPU0)。
source ~/night/common.sh
B=$HOME/batchU; mkdir -p "$B"
exec > "$B/eff_chain.log" 2>&1
touch ~/.gpu_claim_gpu0_batchU; trap 'rm -f ~/.gpu_claim_gpu0_batchU' EXIT

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

# 1) ours dual-L d3,全在线(gram 掩码 + MetaCLIP-L14 分类),env 默认即 shipped 栈
ours() { cd $CC && PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
  GRAM_MCT_DILATE=3 \
  $PY_EVAL eval.py --config ./configs/cfg_ade20k_500.py \
  --cfg-options model.mask_generator=gram model.model_type=ViT-L-14-quickgelu; }

# 2) CorrCLIP-SAM2 在线(官方 AMG 配置 points_per_side=8),同 CLIP 档
corrclip() { cd $CC && PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
  $PY_EVAL eval.py --config ./configs/cfg_ade20k_500.py \
  --cfg-options model.mask_generator=sam2 model.model_type=ViT-L-14-quickgelu; }

# 3) SC-CLIP-L(校准定案配置)
scclip() { cd ~/SC-CLIP && PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
  ~/miniconda3/envs/corrclip/bin/python eval.py \
  --config ./configs/cfg_ade20k_l_500.py --work-dir $B/eff_scclip_wd; }

# 4) CLIPer-L SFSA(SD 2.1-base 在线)
cliper() { cd ~/cliper.code/ovs && PYTHONNOUSERSITE=1 PYTHONPATH=$HOME/cliper.code \
  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 ~/miniconda3/envs/bl/bin/python main_ovs.py \
  --cfg-path ../scripts/config/vit-l-14/ovs_ade150_500.yaml --log-path $B/cliper_log; }

run_timed ours ours
run_timed corrclip_sam2 corrclip
run_timed scclip scclip

# cliper 依赖 SD 2.1 权重 sha256 核验通过(~/sd21_verified.ok 由 sd_verify_then_calib.sh 写),最多等 6h
for i in $(seq 1 360); do [ -f ~/sd21_verified.ok ] && break; sleep 60; done
if [ -f ~/sd21_verified.ok ]; then
  run_timed cliper cliper
else
  echo "SKIP cliper: SD not verified after 6h"
fi
echo "BATCHU_EFF_DONE $(date)"
