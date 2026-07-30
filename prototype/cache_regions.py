"""Cache SigLIP region embeddings + soft GT labels for the ADE train set.

Proposals come from a frozen checkpoint, so embeddings are deterministic ->
cache once, then train adapters on tensors in minutes.

Output: cache_regions/<idx>.pt with {"emb": (K,1152) fp16, "soft": (K,150) fp16}
"""
import argparse, os
import torch
import torch.nn.functional as F

from dino_rtp_seg import DinoRTPSeg
from train_ade import ADESemSeg
from siglip_region import SigLIPRegionClassifier


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ade", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--sig_res", type=int, default=512)
    ap.add_argument("--split", default="training")
    ap.add_argument("--out", default="cache_regions")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    dev = "cuda"
    model = DinoRTPSeg("vits16", args.weights, img_size=640, k=args.k,
                       m=16, use_kernel=True).to(dev).eval()
    ck = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ck["model"], strict=False)
    sig = SigLIPRegionClassifier("./siglip_vision", res=args.sig_res).to(dev)

    ds = ADESemSeg(args.ade, args.split, 640)
    ds.train = False                                     # deterministic resize
    with torch.no_grad(), torch.amp.autocast("cuda", torch.bfloat16):
        for i in range(len(ds)):
            fp = f"{args.out}/{i:06d}.pt"
            if os.path.exists(fp):
                continue
            x, y = ds[i]
            x = x[None].to(dev)
            feat, taps, grid = model.backbone(x)
            masks, regions = model.proposal(feat)
            M = masks[0].float()                          # (K,N)
            emb = sig.region_embeddings(x, M, grid)       # (K,1152)
            yp = F.interpolate(y[None, None].float().to(dev), size=grid,
                               mode="nearest")[0, 0].long().flatten()
            onehot = F.one_hot(yp, 151).float()
            soft = M @ onehot[:, 1:]                      # (K,150)
            soft = soft / soft.sum(-1, keepdim=True).clamp(min=1e-6)
            # also cache the DINOv3 region embeddings for the bridge ensemble
            torch.save({"emb": emb.half().cpu(), "soft": soft.half().cpu(),
                        "reg": regions[0].to(torch.float16).cpu()}, fp)
            if (i + 1) % 500 == 0:
                print(f"{i+1}/{len(ds)}", flush=True)
    print("CACHE_DONE")


if __name__ == "__main__":
    main()
