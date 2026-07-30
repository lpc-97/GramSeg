#!/bin/bash
# ============================================================
# Go/No-Go 实验①: SED-B 冻结 CLIP ConvNeXt-B 主干, 只训 Aggregator
# (decoder/头), 2xGPU(0,1) batch=2, 20k iters, lr 线性缩放 1e-4
# 用法: bash sed_freeze_train_20k.sh   (在服务器上, 任意目录)
# 前置: ~/SED/d2_datasets/coco-stuff/{images,annotations_detectron2}/train2017 就绪
# ============================================================
set -e
cd ~/SED
export DETECTRON2_DATASETS=~/SED/d2_datasets
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0,1
PY=~/miniconda3/envs/bl/bin/python
OUT=~/SED/output/frozenB_20k
mkdir -p "$OUT"

# 说明:
# - CLIP_FINETUNE none  -> sed/sed_model.py __init__ 的 else 分支, CLIP 全部
#   requires_grad=False; 优化器(train_net.py build_optimizer)自动跳过冻结参数
# - BACKBONE_MULTIPLIER 本来就是 0.0 且 backbone 在 __init__ 里被 del, 不用管
# - TEST.EVAL_PERIOD 0: 训练中不 eval (ADE-150 标准 eval 需 POOLING_SIZES [1,1],
#   而它是建模型时的 AvgPool2d, 训练中途改不了; 用 sed_eval_ckpts_ade150.sh 单独跑)
#   注意 d2 的 EvalHook period=0 仍会在第 20000 iter 结束后跑一次 COCO 测试集
#   (coco_2017_test_stuff_all_sem_seg, 171 类, in-domain 指标, ~20min, 白送不亏)
# - CHECKPOINT_PERIOD 2500 -> model_0002499.pth ... model_final.pth 共 8+1 个
$PY train_net.py \
  --config configs/convnextB_768.yaml \
  --num-gpus 2 \
  --dist-url auto \
  --resume \
  OUTPUT_DIR "$OUT" \
  MODEL.SEM_SEG_HEAD.CLIP_FINETUNE none \
  SOLVER.IMS_PER_BATCH 2 \
  SOLVER.BASE_LR 0.0001 \
  SOLVER.MAX_ITER 20000 \
  SOLVER.CHECKPOINT_PERIOD 2500 \
  TEST.EVAL_PERIOD 0 \
  2>&1 | tee -a "$OUT/train_console.log"

# ---------- 显存装不下 768^2 时的 640 降级(先别用, 冻结后 768 大概率能装下) ----------
# SED 的 cost-volume 尺寸由 sed/sed_model.py:94 硬编码的 clip_resolution=(768,768)
# 决定, 只改 INPUT.CROP.SIZE 到 640 不省显存(会被 SIZE_DIVISIBILITY pad 回去再插值到 768)。
# 真降级要 patch 这一行 + 改输入:
#   sed -i 's/self.clip_resolution = (768, 768)/self.clip_resolution = (640, 640)/' ~/SED/sed/sed_model.py
# 然后训练命令追加:
#   INPUT.MIN_SIZE_TRAIN "(640,)" INPUT.CROP.SIZE "(640, 640)" INPUT.SIZE_DIVISIBILITY 640
# 注意: 该 patch 同时影响 eval 前向(eval 也会用 640 clip_resolution), 所以若用降级,
# 必须用同一 patch 重跑官方 ckpt 的 ADE-150 零样本基线, 才能和 20k 结果公平对比
# (31.74 是 768 协议下的基线)。回滚:
#   sed -i 's/self.clip_resolution = (640, 640)/self.clip_resolution = (768, 768)/' ~/SED/sed/sed_model.py
