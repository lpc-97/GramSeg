#!/usr/bin/env bash
# Batch A: CLIP ViT-B/16 main table (7 benchmarks) + OpenAI B/16 ADE fairness row.
# Dual headline config: CorrCLIP framework, DINO-B/8 correlation, pregenerated
# GramSeg masks tag k64_hard_nearest_s896_vits16_ms448_t0.6_pamr10r1024.
# Only CLIP weights change vs the L/14 headline (ViT-L-14-quickgelu -> ViT-B-16-quickgelu).
#
# Usage: GPU=2 bash ~/batchA_vitb16.sh
set -uo pipefail

GPU=${GPU:?set GPU=1 or 2}
EVAL_PY=$HOME/miniconda3/envs/corrclip/bin/python
CORRCLIP=$HOME/CorrCLIP
LOGDIR=$HOME/gram_corrclip/bench/logs
SUMMARY=$HOME/batchA_vitb16_summary.txt
TAG=${TAG:-k64_hard_nearest_s896_vits16_ms448_t0.6_pamr10r1024}
SUFFIX_BASE=${SUFFIX_BASE:-vitb16metaclip}
mkdir -p "$LOGDIR"

maskroot () {
    case "$1" in
        cfg_ade20k) echo ade ;;
        cfg_voc21|cfg_voc20) echo voc ;;
        cfg_coco_stuff164k|cfg_coco_object) echo coco ;;
        cfg_context59|cfg_context60) echo pc ;;
        *) echo UNKNOWN ;;
    esac
}

run_one () {
    local cfg=$1 clip_type=$2 model_type=$3 suffix=$4
    local mask=$CORRCLIP/data/region_masks_gram/$(maskroot "$cfg")/$TAG
    local log=$LOGDIR/eval_${cfg}_${suffix}.log
    if [ "${FORCE:-0}" != 1 ] && grep -qE 'mIoU: *[0-9.]+' "$log" 2>/dev/null; then
        echo "$(date +%F_%T) skip $cfg $suffix (log already has mIoU)" | tee -a "$SUMMARY"
        return 0
    fi
    [ -d "$mask" ] || { echo "$(date +%F_%T) $cfg $suffix FATAL missing mask dir $mask" | tee -a "$SUMMARY"; return 1; }
    cd "$CORRCLIP"
    PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=$GPU "$EVAL_PY" eval.py \
        --config ./configs/${cfg}.py \
        --cfg-options model.clip_type="$clip_type" model.model_type="$model_type" \
                      model.dino_type=dino_vitb8 \
                      model.instance_mask_path="$mask" \
        > "$log" 2>&1
    local miou
    miou=$(grep -oE 'mIoU: *[0-9]+\.[0-9]+' "$log" | tail -1 | grep -oE '[0-9]+\.[0-9]+')
    echo "$(date +%F_%T) $cfg $suffix mIoU=${miou:-FAIL}" | tee -a "$SUMMARY"
}

echo "$(date +%F_%T) === batchA ViT-B/16 start | GPU=$GPU | TAG=$TAG ===" | tee -a "$SUMMARY"

# 1) ADE sanity first (MetaCLIP B/16 fullcc)
run_one cfg_ade20k metaclip_fullcc ViT-B-16-quickgelu "$SUFFIX_BASE"

# 2) OpenAI CLIP B/16 ADE fairness row (weights already in ~/.cache/clip/ViT-B-16.pt)
run_one cfg_ade20k openai ViT-B-16 vitb16openai

# 3) remaining 6 benchmarks, MetaCLIP B/16
for c in cfg_voc21 cfg_voc20 cfg_coco_stuff164k cfg_coco_object cfg_context59 cfg_context60; do
    run_one "$c" metaclip_fullcc ViT-B-16-quickgelu "$SUFFIX_BASE"
done

echo "$(date +%F_%T) === batchA ViT-B/16 DONE ===" | tee -a "$SUMMARY"
