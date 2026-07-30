#!/usr/bin/env bash
# Training-free refinement sweep: {baseline, +multiscale, +pamr, +both} on
# VOC21 (1449 imgs) + ADE (2000 imgs), CorrCLIP ViT-L framework.
#
# For each variant: (a) generate Gram masks with the corresponding switches,
# (b) eval cfg_voc21 + cfg_ade20k with model.instance_mask_path overridden,
# (c) append tagged mIoU lines to results_summary.txt.
#
# Run (server):  GPU=1 bash ~/gram_corrclip/bench/run_refine_sweep.sh
#                GPU=1 bash run_refine_sweep.sh ms pamr   # only these variants
#                FORCE=1 ...   re-run evals whose log already has an mIoU
#
# Decision line: VOC21 >= 68 and ADE not below 26.9 -> Gram mainline revived;
#                VOC21 < 66.5 -> freeze as support chapter.
# Baseline so far: VOC21 64.9 / ADE 26.9 (k64_hard_nearest_s896_vits16).

set -euo pipefail

# ------------------------------ knobs ---------------------------------------
GPU=${GPU:-3}
K=${K:-64}
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
# refinement hyper-params (shared by the variants that use them)
COARSE_SIZE=${COARSE_SIZE:-448}
MERGE_THRESH=${MERGE_THRESH:-0.6}
PAMR_ITERS=${PAMR_ITERS:-10}
PAMR_RES=${PAMR_RES:-0}

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
CORRCLIP=${CORRCLIP:-$HOME/CorrCLIP}
GRAM_GEN=${GRAM_GEN:-$SCRIPT_DIR/../gen_gram_masks.py}
[ -f "$GRAM_GEN" ] || GRAM_GEN=$HOME/gram_corrclip/gen_gram_masks.py

# base TAG identical to run_all_gram.sh so the baseline mask dirs are reused
BASE_TAG=k${K}_${BINARIZE}
[ "$BINARIZE" = topp ] && BASE_TAG=${BASE_TAG}_p${TOP_P}
[ "$CONF" != "0.0" ] && [ "$CONF" != "0" ] && BASE_TAG=${BASE_TAG}_c${CONF}
BASE_TAG=${BASE_TAG}_${UPSAMPLE}_s${IMG_SIZE}_${VARIANT}

LOGDIR=$SCRIPT_DIR/logs
SUMMARY=$SCRIPT_DIR/results_summary.txt
mkdir -p "$LOGDIR"
[ -f "$WEIGHTS" ] || { echo "FATAL: weights not found: $WEIGHTS"; exit 1; }
[ -x "$GEN_PY" ]  || { echo "FATAL: GEN_PY not found: $GEN_PY"; exit 1; }
[ -x "$EVAL_PY" ] || { echo "FATAL: EVAL_PY not found: $EVAL_PY"; exit 1; }
[ -f "$SUMMARY" ] || echo "# dataset  gram_mIoU  [official SAM2_32 ViT-L ref: VOC21 76.7 / VOC20 91.5 / ADE 30.7 / Stuff 34.0 / Object 49.4]" > "$SUMMARY"

VOC_IMGS=$CORRCLIP/data/voc12_val_images
ADE_IMGS=$CORRCLIP/data/ade/ADEChallengeData2016/images/validation

count_files() { ls "$1" 2>/dev/null | grep -c "$2" || true; }

# gen_masks <img_dir> <mask_dir> <expected_count> <extra gen flags...>
gen_masks() {
    local imgs=$1 out=$2 expect=$3; shift 3
    local n_npz; n_npz=$(count_files "$out" '\.npz$')
    if [ "$n_npz" -ge "$expect" ]; then
        echo "[gen] $out complete ($n_npz/$expect), skip"
        return 0
    fi
    echo "[gen] $imgs -> $out ($n_npz/$expect done) flags: $*"
    PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=$GPU "$GEN_PY" "$GRAM_GEN" \
        --images "$imgs" --out "$out" --weights "$WEIGHTS" \
        --variant "$VARIANT" --k "$K" --binarize "$BINARIZE" --top-p "$TOP_P" \
        --conf-thresh "$CONF" --img-size "$IMG_SIZE" --upsample "$UPSAMPLE" \
        "$@"
    n_npz=$(count_files "$out" '\.npz$')
    [ "$n_npz" -ge "$expect" ] || { echo "FATAL: only $n_npz/$expect npz in $out"; exit 1; }
}

# run_eval <cfg_stem> <dataset_label> <mask_dir> <ref_miou> <tag>
run_eval() {
    local cfg=$1 label=$2 maskdir=$3 ref=$4 tag=$5
    local log=$LOGDIR/eval_${cfg}_${tag}.log miou
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
    grep -v "^${label} .*tag=${tag}\b" "$SUMMARY" > "$SUMMARY.tmp" || true
    mv "$SUMMARY.tmp" "$SUMMARY"
    printf '%-10s %-7s (SAM2_32 ref %s)  tag=%s\n' "$label" "$miou" "$ref" "$tag" >> "$SUMMARY"
    echo "[eval] $label mIoU=$miou (ref $ref) tag=$tag"
}

# run_variant <suffix|-> <extra gen flags...>   ('-' = baseline, no suffix)
run_variant() {
    local suffix=$1; shift
    local tag=$BASE_TAG
    [ "$suffix" != "-" ] && tag=${BASE_TAG}_${suffix}
    echo
    echo "=== variant: ${suffix/-/baseline} | TAG=$tag ==="
    local mask_voc=$CORRCLIP/data/region_masks_gram/voc/$tag
    local mask_ade=$CORRCLIP/data/region_masks_gram/ade/$tag
    if [ "$(count_files "$VOC_IMGS" '\.jpg$')" -eq 1449 ]; then
        gen_masks "$VOC_IMGS" "$mask_voc" 1449 "$@"
        run_eval cfg_voc21 VOC21 "$mask_voc" 76.7 "$tag"
    else
        echo "SKIP VOC21: $VOC_IMGS missing/incomplete (run prep_benchmarks.sh)"
    fi
    if [ -d "$ADE_IMGS" ]; then
        gen_masks "$ADE_IMGS" "$mask_ade" 2000 "$@"
        run_eval cfg_ade20k ADE "$mask_ade" 30.7 "$tag"
    else
        echo "SKIP ADE: $ADE_IMGS missing"
    fi
}

# ------------------------------ sweep ----------------------------------------
MS_FLAGS=(--multiscale --coarse-size "$COARSE_SIZE" --merge-thresh "$MERGE_THRESH")
PAMR_FLAGS=(--pamr-iters "$PAMR_ITERS")
[ "$PAMR_RES" != 0 ] && PAMR_FLAGS+=(--pamr-res "$PAMR_RES")

MS_SUFFIX=ms${COARSE_SIZE}_t${MERGE_THRESH}
PAMR_SUFFIX=pamr${PAMR_ITERS}
[ "$PAMR_RES" != 0 ] && PAMR_SUFFIX=${PAMR_SUFFIX}r${PAMR_RES}

VARIANTS=${*:-base ms pamr both}
echo "=== run_refine_sweep | GPU=$GPU | base TAG=$BASE_TAG | variants: $VARIANTS ==="
for v in $VARIANTS; do
    case "$v" in
        base) run_variant - ;;
        ms)   run_variant "$MS_SUFFIX" "${MS_FLAGS[@]}" ;;
        pamr) run_variant "$PAMR_SUFFIX" "${PAMR_FLAGS[@]}" ;;
        both) run_variant "${MS_SUFFIX}_${PAMR_SUFFIX}" "${MS_FLAGS[@]}" "${PAMR_FLAGS[@]}" ;;
        *) echo "unknown variant: $v (use base/ms/pamr/both)"; exit 1 ;;
    esac
done

echo
echo "=== results_summary.txt ==="
cat "$SUMMARY"
echo
echo "Decision line: VOC21 >= 68 & ADE >= 26.9 -> revive mainline; VOC21 < 66.5 -> support chapter."
