"""Cache SED-B cost-map image features + downsampled GT soft labels (COCO-Stuff train).

For each image we store:
  - clip_vis_dense: (640, 24, 24) fp16 — output of the (frozen) CLIP visual
    tower exactly as consumed by Aggregator.correlation at eval time
    (sed/sed_model.py:167 encode_image(clip_images, dense=True)).
  - soft label:     (171, 24, 24) fp16 — per-32x32-block class fraction of the
    detectron2 contiguous-id GT (values sum to the non-ignore fraction <= 1;
    the trainer masks tokens with coverage <= 0.5 and renormalizes).

--ckpt selects whose CLIP visual produces the features:
  ~/offline_staging/sed/sed_convnextB.pth   -> official SED-B (fine-tuned visual)
  ~/SED/output/frozenB_20k/model_final.pth  -> frozen-training ckpt's visual
  none                                      -> raw LAION CLIP (never fine-tuned)

Also dumps text_features.pt once per cache dir: the predictor's precomputed
COCO-171 text embeddings (171, 80, 640) — the exact tensor SED uses in
correlation (LAION text tower; never fine-tuned, unaffected by --ckpt).

Run from the SED repo root (needs sed/, configs/, datasets/coco.json and
DETECTRON2_DATASETS pointing at the dir that contains coco-stuff/).
Resumable: shards already on disk are skipped.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# map raw png values -> 0..170, 255(ignore) -> 171
_LUT = np.full(256, 171, dtype=np.int64)
_LUT[:171] = np.arange(171)


def build_model(config_file: str, ckpt: str):
    from detectron2.config import get_cfg
    from detectron2.projects.deeplab import add_deeplab_config
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.modeling import build_model as d2_build_model
    from sed import add_sed_config

    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_sed_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = ""
    cfg.freeze()
    model = d2_build_model(cfg)          # placed on cfg.MODEL.DEVICE (cuda)
    if ckpt and ckpt.lower() != "none":
        DetectionCheckpointer(model).load(ckpt)
        print(f"[cache] loaded checkpoint {ckpt}")
    else:
        print("[cache] no checkpoint: raw LAION CLIP visual")
    model.eval()
    return cfg, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/convnextB_768.yaml")
    ap.add_argument("--ckpt", required=True,
                    help="SED checkpoint path, or 'none' for raw LAION CLIP")
    ap.add_argument("--num-images", type=int, default=20000)
    ap.add_argument("--shard-size", type=int, default=500)
    ap.add_argument("--out", required=True, help="cache dir, e.g. cache_sed/frozen")
    ap.add_argument("--data-root",
                    default=os.environ.get("DETECTRON2_DATASETS", "datasets"))
    ap.add_argument("--res", type=int, default=768,
                    help="CLIP input resolution (SED eval uses 768)")
    ap.add_argument("--amp", action="store_true",
                    help="fp16 autocast for encoding (default fp32, matches eval)")
    args = ap.parse_args()

    assert args.res % 32 == 0
    grid = args.res // 32                               # 24 for 768
    os.makedirs(args.out, exist_ok=True)

    img_dir = os.path.join(args.data_root, "coco-stuff", "images", "train2017")
    ann_dir = os.path.join(args.data_root, "coco-stuff",
                           "annotations_detectron2", "train2017")
    files = sorted(f for f in os.listdir(img_dir) if f.endswith(".jpg"))
    files = files[: args.num_images]
    n_shards = (len(files) + args.shard_size - 1) // args.shard_size
    print(f"[cache] {len(files)} images -> {n_shards} shards -> {args.out}")

    todo = [s for s in range(n_shards)
            if not os.path.exists(os.path.join(args.out, f"shard_{s * args.shard_size:06d}.pt"))]
    text_path = os.path.join(args.out, "text_features.pt")
    if not todo and os.path.exists(text_path):
        print("[cache] everything cached already, nothing to do")
        return

    cfg, model = build_model(args.config, args.ckpt)
    dev = model.device
    clip_model = model.sem_seg_head.predictor.clip_model
    mean = torch.tensor(cfg.MODEL.CLIP_PIXEL_MEAN, device=dev).view(3, 1, 1)
    std = torch.tensor(cfg.MODEL.CLIP_PIXEL_STD, device=dev).view(3, 1, 1)

    if not os.path.exists(text_path):
        pred = model.sem_seg_head.predictor
        torch.save({"text": pred.text_features.detach().half().cpu(),   # (171,80,640)
                    "classes": pred.class_texts}, text_path)
        print(f"[cache] wrote {text_path} shape {tuple(pred.text_features.shape)}")

    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump({"ckpt": args.ckpt, "config": args.config, "res": args.res,
                   "num_images": len(files), "shard_size": args.shard_size}, f)

    t0, done = time.time(), 0
    with torch.no_grad():
        for s in todo:
            lo = s * args.shard_size
            chunk = files[lo: lo + args.shard_size]
            feats, softs = [], []
            for fn in chunk:
                img = Image.open(os.path.join(img_dir, fn)).convert("RGB")
                x = torch.from_numpy(np.asarray(img).copy()).to(dev)
                x = x.permute(2, 0, 1).float()
                x = (x - mean) / std
                x = F.interpolate(x[None], size=(args.res, args.res),
                                  mode="bilinear", align_corners=False)
                with torch.autocast("cuda", torch.float16, enabled=args.amp):
                    feat = clip_model.encode_image(x, dense=True)["clip_vis_dense"]
                feats.append(feat[0].float().half().cpu())          # (640,g,g)

                ann = Image.open(os.path.join(ann_dir, fn[:-4] + ".png"))
                ann = ann.resize((args.res, args.res), Image.NEAREST)
                y = torch.from_numpy(_LUT[np.asarray(ann)]).to(dev)  # (res,res)
                onehot = F.one_hot(y, 172).permute(2, 0, 1).float()  # (172,res,res)
                frac = F.avg_pool2d(onehot[None], kernel_size=32)[0, :171]
                softs.append(frac.half().cpu())                      # (171,g,g)
                done += 1

            out_fp = os.path.join(args.out, f"shard_{lo:06d}.pt")
            tmp_fp = out_fp + ".tmp"
            torch.save({"feats": torch.stack(feats),
                        "softs": torch.stack(softs),
                        "files": chunk}, tmp_fp)
            os.replace(tmp_fp, out_fp)
            rate = done / max(time.time() - t0, 1e-6)
            print(f"[cache] shard {s + 1}/{n_shards} ({done} imgs, "
                  f"{rate:.1f} img/s, eta {(len(todo) * args.shard_size - done) / rate / 60:.1f} min)",
                  flush=True)
    print(f"CACHE_DONE {time.time() - t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
