#!/usr/bin/env bash
# Prepare VOC2012 / COCO-Stuff164k / COCO-Object data layouts for CorrCLIP.
# Idempotent: every step checks before doing work; safe to re-run (e.g. run
# again after val2017_hf.zip finishes downloading).
#
# Layouts required (verified against configs/ and custom_datasets.py):
#   data/VOC2012/{JPEGImages,SegmentationClass,ImageSets/Segmentation/val.txt}
#     - cfg_voc20.py + cfg_voc21.py, SegmentationClass palette pngs used AS-IS
#   data/coco/images/val2017/*.jpg
#   data/coco/annotations/val2017/*_labelTrainIds.png     (COCOStuffDataset)
#   data/coco/annotations/val2017/*_instanceTrainIds.png  (COCOObjectDataset)
#     - both converted from the ORIGINAL stuffthingmaps in labroot;
#       instances_trainval2017.zip is NOT needed (cvt_coco_object.py works
#       off stuffthingmaps pngs, not the instances json).
#
# Run (server): bash ~/gram_corrclip/bench/prep_benchmarks.sh

set -euo pipefail

CORRCLIP=${CORRCLIP:-$HOME/CorrCLIP}
STAGE=${STAGE:-$HOME/offline_staging}
LABROOT=${LABROOT:-$HOME/dino_rtp_seg/data/coco/labroot/val2017}
PY=${PY:-$HOME/miniconda3/envs/corrclip/bin/python}   # needs numpy + PIL only
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

mkdir -p "$CORRCLIP/data"

count() { ls "$1" 2>/dev/null | grep -c "$2" || true; }

# ============================ 1. VOC2012 ====================================
echo "== [VOC2012] =="
VOC_TAR=$STAGE/benchmarks/VOCtrainval_11-May-2012.tar
VOC_ROOT=$STAGE/extracted/VOCdevkit/VOC2012

if [ ! -d "$VOC_ROOT/JPEGImages" ]; then
    [ -f "$VOC_TAR" ] || { echo "SKIP VOC: tar not found: $VOC_TAR"; VOC_OK=0; }
    if [ -f "$VOC_TAR" ]; then
        echo "extracting $VOC_TAR -> $STAGE/extracted"
        mkdir -p "$STAGE/extracted"
        tar -xf "$VOC_TAR" -C "$STAGE/extracted"
    fi
fi

if [ -d "$VOC_ROOT/JPEGImages" ]; then
    [ -e "$CORRCLIP/data/VOC2012" ] || ln -s "$VOC_ROOT" "$CORRCLIP/data/VOC2012"
    VAL_TXT=$CORRCLIP/data/VOC2012/ImageSets/Segmentation/val.txt
    N_VAL=$(grep -c . "$VAL_TXT")
    [ "$N_VAL" -eq 1449 ] || { echo "FATAL: val.txt has $N_VAL entries, expected 1449"; exit 1; }
    N_SEG=$(count "$CORRCLIP/data/VOC2012/SegmentationClass" '\.png$')
    echo "VOC2012 linked: val.txt=$N_VAL, SegmentationClass pngs=$N_SEG (>=1449 needed)"

    # val-only image link farm, so gen_gram_masks.py only touches the 1449
    # val images instead of all 17125 JPEGImages
    VAL_LINKS=$CORRCLIP/data/voc12_val_images
    if [ "$(count "$VAL_LINKS" '\.jpg$')" -ne 1449 ]; then
        echo "building val-only image links -> $VAL_LINKS"
        mkdir -p "$VAL_LINKS"
        while IFS= read -r id; do
            [ -n "$id" ] && ln -sf "$VOC_ROOT/JPEGImages/$id.jpg" "$VAL_LINKS/$id.jpg"
        done < "$VAL_TXT"
        [ "$(count "$VAL_LINKS" '\.jpg$')" -eq 1449 ] || { echo "FATAL: val link farm incomplete"; exit 1; }
    fi
    echo "VOC2012 ready."
fi

# ============================ 2. COCO images ================================
echo "== [COCO val2017 images] =="
COCO_IMG_LINK=$CORRCLIP/data/coco/images/val2017

if [ "$(count "$COCO_IMG_LINK/" '\.jpg$')" -eq 5000 ]; then
    echo "COCO images already in place."
else
    ZIP=$STAGE/coco/val2017_hf.zip
    [ -f "$ZIP" ] || ZIP=$(ls "$STAGE"/coco/val2017*.zip 2>/dev/null | head -1 || true)
    if [ -z "${ZIP:-}" ] || [ ! -f "$ZIP" ]; then
        echo "SKIP COCO images: no val2017 zip under $STAGE/coco (still downloading?)"
    else
        EXTRACT=$STAGE/coco/extracted_val2017
        mkdir -p "$EXTRACT"
        echo "unzipping $ZIP -> $EXTRACT (skips existing files)"
        unzip -q -n "$ZIP" -d "$EXTRACT"
        # layout A: zip contains val2017/*.jpg ; layout B: bare jpgs
        if [ -d "$EXTRACT/val2017" ]; then SRC=$EXTRACT/val2017
        else
            SRC=$(find "$EXTRACT" -maxdepth 3 -type d -name val2017 | head -1)
            [ -n "$SRC" ] || SRC=$EXTRACT
        fi
        N_JPG=$(count "$SRC" '\.jpg$')
        [ "$N_JPG" -eq 5000 ] || { echo "FATAL: found $N_JPG jpgs under $SRC, expected 5000 (zip incomplete?)"; exit 1; }
        mkdir -p "$CORRCLIP/data/coco/images"
        [ -e "$COCO_IMG_LINK" ] || ln -s "$SRC" "$COCO_IMG_LINK"
        echo "COCO images ready: $COCO_IMG_LINK -> $SRC ($N_JPG jpgs)"
    fi
fi

# ============================ 3. COCO labels ================================
echo "== [COCO annotations: stuff _labelTrainIds + object _instanceTrainIds] =="
ANN=$CORRCLIP/data/coco/annotations/val2017
[ -d "$LABROOT" ] || { echo "FATAL: labroot not found: $LABROOT"; exit 1; }
N_SRC=$(count "$LABROOT" '\.png$')
[ "$N_SRC" -eq 5000 ] || echo "WARN: labroot has $N_SRC pngs (expected 5000)"

N_STUFF=$(count "$ANN" '_labelTrainIds\.png$')
N_OBJ=$(count "$ANN" '_instanceTrainIds\.png$')
if [ "$N_STUFF" -eq 5000 ] && [ "$N_OBJ" -eq 5000 ]; then
    echo "COCO annotations already converted ($N_STUFF stuff / $N_OBJ object)."
else
    echo "converting labroot -> $ANN ($N_STUFF/$N_OBJ present)"
    mkdir -p "$ANN"
    PYTHONNOUSERSITE=1 "$PY" "$SCRIPT_DIR/convert_coco_labels.py" \
        --src "$LABROOT" --dst "$ANN"
    N_STUFF=$(count "$ANN" '_labelTrainIds\.png$')
    N_OBJ=$(count "$ANN" '_instanceTrainIds\.png$')
    [ "$N_STUFF" -eq 5000 ] && [ "$N_OBJ" -eq 5000 ] \
        || { echo "FATAL: conversion incomplete ($N_STUFF/$N_OBJ)"; exit 1; }
    echo "COCO annotations ready."
fi

# ============================ 4. missing datasets ===========================
echo "== [Pascal Context / Cityscapes] =="
echo "SKIP: data missing. When available, wire as:"
echo "  PC:         data/pcontext  (VOC2010 JPEGImages + trainval_merged.json,"
echo "              cfg_context59.py / cfg_context60.py, needs Detail API conversion)"
echo "  Cityscapes: data/cityscapes/{leftImg8bit,gtFine}/val (cfg_city_scapes.py)"

echo
echo "== prep done =="
for d in "$CORRCLIP/data/VOC2012" "$CORRCLIP/data/voc12_val_images" \
         "$CORRCLIP/data/coco/images/val2017" "$CORRCLIP/data/coco/annotations/val2017"; do
    printf '  %-55s %s\n' "$d" "$( [ -e "$d" ] && echo OK || echo MISSING )"
done
