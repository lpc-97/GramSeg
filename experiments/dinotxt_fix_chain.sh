#!/bin/bash
# 等 GPU1 夜链(Batch V 后半)完 → 修正版 dinotxt 校准:ADE 512 官方 + City 512 官方 + ADE 我们协议
exec > ~/dinotxt_fix.log 2>&1
for i in $(seq 1 600); do grep -q "NIGHT_GPU1_DONE" ~/night_gpu1.log 2>/dev/null && break; sleep 60; done
HEAD_SNAP=$(find ${CACHE_ROOT:-$HOME/cache}/dinotxt_cache -name "dinov3_vitl16_dinotxt_vision_head_and_text_encoder.pth" | head -1)
BPE=$(find ${CACHE_ROOT:-$HOME/cache}/dinotxt_cache -name "bpe_simple_vocab_16e6.txt.gz" | head -1)
BBW=~/dino_rtp_seg/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
dt() { tag=$1; shift; PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=1 ~/miniconda3/envs/bl/bin/python \
  ~/dinotxt_ade_eval.py --head-weights "$HEAD_SNAP" --backbone-weights "$BBW" --bpe "$BPE" "$@" \
  > ~/batchU/dinotxt_$tag.log 2>&1; echo "dinotxt_$tag rc=$? $(grep FINAL ~/batchU/dinotxt_$tag.log)"; }
dt fix_ade512 --mode official --resize 512 --names ~/nb_ade_names.txt
dt fix_city512 --mode official --dataset city --resize 512
dt fix_ours_ade --mode ours
echo "DINOTXT_FIX_DONE $(date)"
