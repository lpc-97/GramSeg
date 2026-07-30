"""Zero-shot qualitative check: Gram-Seed region masks on real images.

No training involved -- validates that frozen DINOv3 affinity alone yields
object-level grouping (contribution C2). Saves a seed-region overlay PNG.

Usage: python run_visualize.py --weights dinov3_vits16.safetensors --image test1.jpg
"""
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from dino_rtp_seg import DinoRTPSeg

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--variant", default="vits16")
    ap.add_argument("--image", required=True)
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = DinoRTPSeg(args.variant, args.weights, img_size=args.size,
                       k=args.k).to(dev).eval()

    img = Image.open(args.image).convert("RGB").resize((args.size, args.size))
    x = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    x = ((x - MEAN) / STD)[None].to(dev)

    with torch.no_grad():
        feat, taps, grid = model.backbone(x)
        masks, regions = model.proposal(feat)          # (1,K,N)
    h, w = grid
    hard = masks[0].argmax(0).reshape(h, w).cpu().numpy()   # each patch -> region id

    # also run a point-prompt demo at image center
    with torch.no_grad():
        out = model(x, points=torch.tensor([[0.5, 0.5]], device=dev))
        pm = torch.sigmoid(out["mask_hi"][0]).cpu().numpy()

    rng = np.random.RandomState(0)
    palette = rng.randint(0, 255, (args.k, 3), dtype=np.uint8)
    seg = palette[hard]
    seg = np.array(Image.fromarray(seg).resize(img.size, Image.NEAREST))
    overlay = (0.55 * np.array(img) + 0.45 * seg).astype(np.uint8)

    pmask = (pm - pm.min()) / (np.ptp(pm) + 1e-6)
    pmask = np.array(Image.fromarray((pmask * 255).astype(np.uint8)).resize(img.size))
    heat = np.zeros_like(np.array(img)); heat[..., 0] = pmask
    pview = (0.6 * np.array(img) + 0.4 * heat).astype(np.uint8)

    canvas = np.concatenate([np.array(img), overlay, pview], axis=1)
    out_path = args.out or (args.image.rsplit(".", 1)[0] + "_gramseed.png")
    Image.fromarray(canvas).save(out_path)
    n_used = len(np.unique(hard))
    print(f"saved {out_path} | active regions {n_used}/{args.k} | grid {h}x{w}")


if __name__ == "__main__":
    main()
