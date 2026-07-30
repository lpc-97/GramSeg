"""Cache SigLIP region embeddings + soft GT labels on COCO-Stuff 10k.

Zero-shot discipline: the adapter trained on this cache sees ONLY COCO
supervision; ADE evaluation stays zero-shot.
"""
import argparse, os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from dino_rtp_seg import DinoRTPSeg
from train_ade import MEAN, STD
from coco_names import CLS_TO_TRAIN
from siglip_region import SigLIPRegionClassifier

_LUT = np.full(256, 0, dtype=np.int64)
for raw, tr in CLS_TO_TRAIN.items():
    if raw != 255:
        _LUT[raw] = tr + 1                                # 1..171, 0=ignore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--sig_res", type=int, default=512)
    ap.add_argument("--out", default="cache_h")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    dev = "cuda"
    model = DinoRTPSeg("vits16", args.weights, img_size=640, k=args.k,
                       m=16, use_kernel=True).to(dev).eval()
    ck = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ck["model"], strict=False)
    sig = SigLIPRegionClassifier("./siglip_vision", res=args.sig_res).to(dev)

    img_dir = os.path.join(args.coco, "images")
    lab_dir = os.path.join(args.coco, "labels")
    files = sorted(os.listdir(img_dir))
    with torch.no_grad(), torch.amp.autocast("cuda", torch.bfloat16):
        for i, fn in enumerate(files):
            fp = f"{args.out}/{i:06d}.pt"
            if os.path.exists(fp):
                continue
            img = Image.open(os.path.join(img_dir, fn)).convert("RGB").resize((640, 640), Image.BILINEAR)
            ann = Image.open(os.path.join(lab_dir, fn.replace(".jpg", ".png"))).resize((640, 640), Image.NEAREST)
            x = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
            x = ((x - MEAN) / STD)[None].to(dev)
            y = torch.from_numpy(_LUT[np.array(ann)]).to(dev)

            feat, taps, grid = model.backbone(x)
            masks, regions = model.proposal(feat)
            M = masks[0].float()
            emb = sig.region_embeddings(x, M, grid)
            yp = F.interpolate(y[None, None].float(), size=grid,
                               mode="nearest")[0, 0].long().flatten()
            onehot = F.one_hot(yp, 172).float()            # 0=ignore, 1..171
            soft = M @ onehot[:, 1:]
            soft = soft / soft.sum(-1, keepdim=True).clamp(min=1e-6)
            torch.save({"emb": emb.half().cpu(), "soft": soft.half().cpu()}, fp)
            if (i + 1) % 500 == 0:
                print(f"{i+1}/{len(files)}", flush=True)
    print("CACHE_DONE")


if __name__ == "__main__":
    main()
