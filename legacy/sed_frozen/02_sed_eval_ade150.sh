#!/bin/bash
# SED-B 零样本 ADE-150 eval 复现,目标锁定 31.8 mIoU
# 前置:detectron2 已装(01 脚本);SED 仓库在 ~/SED;SED-B ckpt 已 scp 到 ~/SED/weights/sed_convnextB.pth
# GPU0 常忙 → 只用 1/2/3
set -e
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh
conda activate bl
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=1,2,3

cd ~/SED
# 依赖补齐检查(ftfy/regex/einops/timm;env bl 大概率已有 timm)
python -c "import ftfy, regex, einops, timm; print('deps OK')"

# 数据:SED 期望 $DETECTRON2_DATASETS/ADEChallengeData2016/{images,annotations}/validation
# 服务器已有 ADE20K —— 用软链适配目录布局,不重复占盘:
export DETECTRON2_DATASETS=~/d2_datasets
mkdir -p $DETECTRON2_DATASETS
# TODO 按服务器实际 ADE 路径改这一行:
# ln -sfn /path/to/ADEChallengeData2016 $DETECTRON2_DATASETS/ADEChallengeData2016
python datasets/prepare_ade20k_150.py   # 生成 annotations_detectron2/validation

# eval(3 GPU;eval.sh 第二参是 GPU 数)
sh eval.sh configs/convnextB_768.yaml 3 output/sed_b_eval MODEL.WEIGHTS weights/sed_convnextB.pth 2>&1 | tee ~/sed_b_eval.log
grep -E "mIoU|copypaste" ~/sed_b_eval.log | tail -5
# 期望:ADE-150 mIoU ≈ 31.8(fast 版 31.6)
