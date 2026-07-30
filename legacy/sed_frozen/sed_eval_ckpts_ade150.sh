#!/bin/bash
# ============================================================
# 对 frozenB_20k 输出目录里所有 checkpoint 逐个跑 ADE-150 零样本 eval
# 协议与官方 eval.sh / 你复现 31.74 的完全一致:
#   ade150.json + ade20k_150_test_sem_seg + POOLING_SIZES [1,1]
# 用法: bash sed_eval_ckpts_ade150.sh [GPU_ID]   (默认 GPU 0)
#   - 训练结束后跑: bash sed_eval_ckpts_ade150.sh 0
#   - 想训练期间蹭卡: 先 nvidia-smi 确认目标卡余量 >10GB 再跑, OOM 就等训完
# 幂等: 已出 copypaste 结果的 ckpt 自动跳过, 可反复重跑(配合训练中途新 ckpt)
# ============================================================
set -e
cd ~/SED
export DETECTRON2_DATASETS=~/SED/d2_datasets
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=${1:-0}
PY=~/miniconda3/envs/bl/bin/python
OUT=~/SED/output/frozenB_20k

shopt -s nullglob
for ckpt in "$OUT"/model_*.pth; do
  tag=$(basename "$ckpt" .pth)
  ed=$OUT/eval-ade150-$tag
  if grep -q copypaste "$ed/log.txt" 2>/dev/null; then
    echo "[skip] $tag already evaluated"
    continue
  fi
  echo "[eval] $tag"
  $PY train_net.py \
    --config configs/convnextB_768.yaml \
    --num-gpus 1 \
    --dist-url auto \
    --eval-only \
    OUTPUT_DIR "$ed" \
    MODEL.WEIGHTS "$ckpt" \
    MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON datasets/ade150.json \
    DATASETS.TEST '("ade20k_150_test_sem_seg",)' \
    MODEL.SEM_SEG_HEAD.POOLING_SIZES '[1,1]'
done

echo "================ ADE-150 mIoU 汇总 ================"
grep -H copypaste "$OUT"/eval-ade150-*/log.txt || echo "no results yet"
