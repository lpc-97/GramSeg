#!/usr/bin/env bash
# Multi-benchmark eval: CorrCLIP ViT-L + Gram-Seed masks.
# For each available benchmark: (a) generate Gram masks for the val images,
# (b) run eval.py with mask_generator=None (already the base_config default)
# and model.instance_mask_path overridden, (c) append mIoU to results_summary.txt.
#
# Prereq: bash prep_benchmarks.sh   (run once; re-run after downloads finish)
# Run:    GPU=1 bash ~/gram_corrclip/bench/run_all_gram.sh
#         GPU=1 bash run_all_gram.sh --with-ade    # also re-eval ADE
#         FORCE=1 ... to re-run evals whose log already has an mIoU
#
# Official SAM2_32 ViT-L reference: VOC21 76.7 / VOC20 91.5 / ADE 30.7 /
# Stuff 34.0 / Object 49.4.  Our Gram-ADE so far: 26.87 (k64 hard nearest 896).

set -euo pipefail

# ------------------------------ knobs ---------------------------------------
GPU=${GPU:-3}
K=${K:-64}
CLUSTER=${CLUSTER:-gram}     # gram | kmeans (spherical, same feats) | slic (RGB superpixels)
SKIP_COCO=${SKIP_COCO:-0}    # 1 = only VOC (+ADE with --with-ade), e.g. baseline runs
BINARIZE=${BINARIZE:-hard}
TOP_P=${TOP_P:-0.9}
CONF=${CONF:-0.0}
IMG_SIZE=${IMG_SIZE:-896}
UPSAMPLE=${UPSAMPLE:-nearest}
VARIANT=${VARIANT:-vits16}
WEIGHTS=${WEIGHTS:-$HOME/dino_rtp_seg/dinov3_vits16.safetensors}
GEN_PY=${GEN_PY:-$HOME/miniconda3/envs/bl/bin/python}
EVAL_PY=${EVAL_PY:-$HOME/miniconda3/envs/corrclip/bin/python}
FORCE=${FORCE:-0}
SKIP_ADE=${SKIP_ADE:-1}
for a in "$@"; do
    case "$a" in
        --with-ade) SKIP_ADE=0 ;;
        --skip-ade) SKIP_ADE=1 ;;
        *) echo "unknown arg: $a (use --with-ade / --skip-ade)"; exit 1 ;;
    esac
done

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
CORRCLIP=${CORRCLIP:-$HOME/CorrCLIP}
GRAM_GEN=${GRAM_GEN:-$SCRIPT_DIR/../gen_gram_masks.py}
[ -f "$GRAM_GEN" ] || GRAM_GEN=$HOME/gram_corrclip/gen_gram_masks.py

# TAG identical to run_gram_eval.sh so the existing ADE mask dir is reused
TAG=k${K}_${BINARIZE}
[ "$BINARIZE" = topp ] && TAG=${TAG}_p${TOP_P}
[ "$CONF" != "0.0" ] && [ "$CONF" != "0" ] && TAG=${TAG}_c${CONF}
TAG=${TAG}_${UPSAMPLE}_s${IMG_SIZE}_${VARIANT}
[ "$CLUSTER" != gram ] && TAG=${CLUSTER}_${TAG}   # baseline runs get own dirs/logs

LOGDIR=$SCRIPT_DIR/logs
SUMMARY=$SCRIPT_DIR/results_summary.txt
mkdir -p "$LOGDIR"
echo "=== run_all_gram | TAG=$TAG | GPU=$GPU ==="
# slic is image-only: no DINOv3 weights needed
[ "$CLUSTER" = slic ] || [ -f "$WEIGHTS" ] || { echo "FATAL: weights not found: $WEIGHTS"; exit 1; }
[ -x "$GEN_PY" ]  || { echo "FATAL: GEN_PY not found: $GEN_PY"; exit 1; }
[ -x "$EVAL_PY" ] || { echo "FATAL: EVAL_PY not found: $EVAL_PY"; exit 1; }

if [ ! -f "$SUMMARY" ]; then
    {
        echo "# dataset  gram_mIoU  [official SAM2_32 ViT-L ref: VOC21 76.7 / VOC20 91.5 / ADE 30.7 / Stuff 34.0 / Object 49.4]"
    } > "$SUMMARY"
fi

count_files() { ls "$1" 2>/dev/null | grep -c "$2" || true; }

# ------------------------- step a: mask generation --------------------------
# gen_masks <img_dir> <mask_dir> <expected_count>
gen_masks() {
    local imgs=$1 out=$2 expect=$3
    local n_npz; n_npz=$(count_files "$out" '\.npz$')
    if [ "$n_npz" -ge "$expect" ]; then
        echo "[gen] $out complete ($n_npz/$expect), skip"
        return 0
    fi
    echo "[gen] $imgs -> $out ($n_npz/$expect done)"
    PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=$GPU "$GEN_PY" "$GRAM_GEN" \
        --images "$imgs" --out "$out" --weights "$WEIGHTS" \
        --variant "$VARIANT" --k "$K" --binarize "$BINARIZE" --top-p "$TOP_P" \
        --conf-thresh "$CONF" --img-size "$IMG_SIZE" --upsample "$UPSAMPLE" \
        --cluster "$CLUSTER"
    n_npz=$(count_files "$out" '\.npz$')
    [ "$n_npz" -ge "$expect" ] || { echo "FATAL: only $n_npz/$expect npz in $out"; exit 1; }
}

# ------------------------- step b: eval + summary ---------------------------
# run_eval <cfg_stem> <dataset_label> <mask_dir> <ref_miou>
run_eval() {
    local cfg=$1 label=$2 maskdir=$3 ref=$4
    local log=$LOGDIR/eval_${cfg}_${TAG}.log miou
    if [ "$FORCE" != 1 ] && [ -s "$log" ] && grep -qE 'mIoU: *[0-9]+\.[0-9]+' "$log"; then
        echo "[eval] $cfg: reuse existing $log (FORCE=1 to re-run)"
    else
        echo "[eval] $cfg (ViT-L, masks=$maskdir) -> $log"
        ( cd "$CORRCLIP" && \
          PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=$GPU "$EVAL_PY" eval.py \
            --config "./configs/${cfg}.py" \
            --cfg-options model.model_type=ViT-L-14-quickgelu \
                          model.instance_mask_path="$maskdir" \
          2>&1 | tee "$log" )
    fi
    miou=$(grep -oE 'mIoU: *[0-9]+\.[0-9]+' "$log" | tail -1 | grep -oE '[0-9]+\.[0-9]+' || true)
    [ -n "$miou" ] || { echo "FATAL: no mIoU parsed from $log"; exit 1; }
    # replace previous line for the same dataset+TAG, then append
    grep -v "^${label} .*tag=${TAG}\b" "$SUMMARY" > "$SUMMARY.tmp" || true
    mv "$SUMMARY.tmp" "$SUMMARY"
    printf '%-10s %-7s (SAM2_32 ref %s)  tag=%s\n' "$label" "$miou" "$ref" "$TAG" >> "$SUMMARY"
    echo "[eval] $label mIoU=$miou (ref $ref)"
}

# ------------------------------ benchmarks ----------------------------------
VOC_IMGS=$CORRCLIP/data/voc12_val_images
COCO_IMGS=$CORRCLIP/data/coco/images/val2017
COCO_ANN=$CORRCLIP/data/coco/annotations/val2017
ADE_IMGS=$CORRCLIP/data/ade/ADEChallengeData2016/images/validation
MASK_VOC=$CORRCLIP/data/region_masks_gram/voc/$TAG
MASK_COCO=$CORRCLIP/data/region_masks_gram/coco/$TAG
MASK_ADE=$CORRCLIP/data/region_masks_gram/ade/$TAG

# --- VOC (voc20 + voc21 share the same image set and mask dir) ---
if [ "$(count_files "$VOC_IMGS" '\.jpg$')" -eq 1449 ]; then
    gen_masks "$VOC_IMGS" "$MASK_VOC" 1449
    run_eval cfg_voc21 VOC21 "$MASK_VOC" 76.7
    run_eval cfg_voc20 VOC20 "$MASK_VOC" 91.5
else
    echo "SKIP VOC: $VOC_IMGS missing/incomplete (run prep_benchmarks.sh)"
fi

# --- COCO (stuff + object share images and mask dir) ---
if [ "$SKIP_COCO" = 1 ]; then
    echo "SKIP COCO (SKIP_COCO=1)"
elif [ "$(count_files "$COCO_IMGS/" '\.jpg$')" -eq 5000 ]; then
    gen_masks "$COCO_IMGS" "$MASK_COCO" 5000
    if [ "$(count_files "$COCO_ANN" '_labelTrainIds\.png$')" -eq 5000 ]; then
        run_eval cfg_coco_stuff164k Stuff "$MASK_COCO" 34.0
    else
        echo "SKIP Stuff: _labelTrainIds pngs missing (run prep_benchmarks.sh)"
    fi
    if [ "$(count_files "$COCO_ANN" '_instanceTrainIds\.png$')" -eq 5000 ]; then
        run_eval cfg_coco_object Object "$MASK_COCO" 49.4
    else
        echo "SKIP Object: _instanceTrainIds pngs missing (run prep_benchmarks.sh)"
    fi
else
    echo "SKIP COCO: $COCO_IMGS missing/incomplete (zip still downloading? run prep_benchmarks.sh)"
fi

# --- ADE (already done: 26.87 with k64_hard_nearest_s896_vits16) ---
if [ "$SKIP_ADE" = 1 ]; then
    echo "SKIP ADE (known: gram 26.87 / SAM2_32 30.71); use --with-ade to re-run"
elif [ -d "$ADE_IMGS" ]; then
    gen_masks "$ADE_IMGS" "$MASK_ADE" 2000
    run_eval cfg_ade20k ADE "$MASK_ADE" 30.7
else
    echo "SKIP ADE: $ADE_IMGS missing"
fi

echo
echo "=== results_summary.txt ==="
cat "$SUMMARY"
