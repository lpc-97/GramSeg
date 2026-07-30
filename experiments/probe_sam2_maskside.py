#!/usr/bin/env python
# SAM2 掩码端受控计时(CorrCLIP shipped 配置:pps=8, iou/stab 0.4, fp16, multimask off)
# 口径对齐旧 gram 探针:ADE val 前 200 图,mask 生成段独立计时(不含分类/相关性),含 10 图 warmup。
import os, sys, time, glob
import numpy as np
import torch
from PIL import Image

os.chdir(os.path.expanduser("~/CorrCLIP"))
sys.path.insert(0, os.path.expanduser("~/CorrCLIP"))
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

sam2 = build_sam2("sam2_hiera_l.yaml", "sam2_hiera_large.pt", device="cuda", apply_postprocessing=False)
sam2 = sam2.half()
gen = SAM2AutomaticMaskGenerator(model=sam2, points_per_side=8, pred_iou_thresh=0.4,
                                 stability_score_thresh=0.4, multimask_output=False)

files = sorted(glob.glob(os.path.expanduser(
    "~/dino_rtp_seg/data/ADEChallengeData2016/images/validation/*.jpg")))[:210]
times = []
for i, fp in enumerate(files):
    img = np.array(Image.open(fp).convert("RGB"))
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        masks = gen.generate(img)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    if i >= 10:  # warmup 10
        times.append(dt)
    if (i + 1) % 50 == 0:
        print(f"[{i+1}] mean={np.mean(times)*1000:.1f}ms n_masks={len(masks)}", flush=True)

print(f"SAM2_MASKSIDE_FINAL mean={np.mean(times)*1000:.1f}ms median={np.median(times)*1000:.1f}ms "
      f"n={len(times)} peak_mem={torch.cuda.max_memory_allocated()/2**30:.2f}GB", flush=True)
