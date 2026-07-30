#!/usr/bin/env python
"""Convert ORIGINAL COCO stuffthingmaps val2017 pngs into the two label
formats CorrCLIP's configs expect, in one pass (val split only, 5000 pngs).

Source pngs (~/dino_rtp_seg/data/coco/labroot/val2017/*.png) are the official
stuffthingmaps_trainval2017: pixel = 0-indexed cocostuff id 0..181 (with 11
unused thing ids missing: 11,25,28,29,44,65,67,68,70,82,90), 255 = unlabeled.

Outputs written next to each other into --dst:
  <stem>_labelTrainIds.png     for COCOStuffDataset (mmseg builtin,
                               seg_map_suffix='_labelTrainIds.png',
                               reduce_zero_label=False):
                               valid ids -> contiguous 0..170, 255 -> 255.
                               Identical to mmseg tools/dataset_converters/
                               coco_stuff164k.py.
  <stem>_instanceTrainIds.png  for COCOObjectDataset (custom_datasets.py:86,
                               seg_map_suffix='_instanceTrainIds.png',
                               reduce_zero_label=False):
                               thing ids (<=90) -> 1..80,
                               stuff ids (>90) and 255 -> 0 (background).
                               Identical to datasets/cvt_coco_object.py
                               (which needs train+val = 123287 pngs and would
                               assert-fail on val-only, hence this script).

Usage:
  python convert_coco_labels.py --src ~/dino_rtp_seg/data/coco/labroot/val2017 \
                                --dst ~/CorrCLIP/data/coco/annotations/val2017
"""
import argparse
import os
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

REMOVED = {11, 25, 28, 29, 44, 65, 67, 68, 70, 82, 90}
VALID = [i for i in range(182) if i not in REMOVED]          # 171 ids
assert len(VALID) == 171

# stuff LUT: contiguous train ids 0..170, everything else / 255 -> 255
LUT_STUFF = np.full(256, 255, dtype=np.uint8)
for tr, cid in enumerate(VALID):
    LUT_STUFF[cid] = tr

# object LUT: thing ids -> 1..80, stuff / unlabeled / removed -> 0
LUT_OBJ = np.zeros(256, dtype=np.uint8)
THINGS = [i for i in VALID if i <= 90]                        # 80 ids
assert len(THINGS) == 80
for tr, cid in enumerate(THINGS):
    LUT_OBJ[cid] = tr + 1


def convert_one(png, dst, force=False):
    stem = png.stem
    out_stuff = dst / f"{stem}_labelTrainIds.png"
    out_obj = dst / f"{stem}_instanceTrainIds.png"
    if not force and out_stuff.exists() and out_obj.exists():
        return 0
    arr = np.array(Image.open(png))
    if arr.ndim == 3:                       # paranoid: some tools save RGB
        arr = arr[..., 0]
    arr = arr.astype(np.uint8)
    Image.fromarray(LUT_STUFF[arr]).save(out_stuff)
    Image.fromarray(LUT_OBJ[arr]).save(out_obj)
    return 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, help="dir with original stuffthingmaps val2017 pngs")
    ap.add_argument("--dst", required=True, help="output dir (data/coco/annotations/val2017)")
    ap.add_argument("--nproc", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    src, dst = Path(args.src).expanduser(), Path(args.dst).expanduser()
    dst.mkdir(parents=True, exist_ok=True)
    pngs = sorted(p for p in src.iterdir()
                  if p.suffix == ".png" and "TrainIds" not in p.name)
    assert pngs, f"no source pngs under {src}"
    print(f"{len(pngs)} source pngs, converting with {args.nproc} procs -> {dst}")

    with Pool(args.nproc) as pool:
        done = sum(pool.map(partial(convert_one, dst=dst, force=args.force),
                            pngs, chunksize=64))
    print(f"converted {done}, skipped {len(pngs) - done} (already present)")

    # quick self-check on one converted file
    chk = np.array(Image.open(dst / f"{pngs[0].stem}_labelTrainIds.png"))
    vals = np.unique(chk)
    assert vals.max() <= 255 and ((vals <= 170) | (vals == 255)).all(), vals
    print(f"self-check OK: {pngs[0].stem}_labelTrainIds.png ids={vals[:10]}...")


if __name__ == "__main__":
    main()
