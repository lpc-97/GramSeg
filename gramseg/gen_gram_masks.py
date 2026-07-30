#!/usr/bin/env python
"""Offline route (a): pre-generate Gram-Seed region masks for CorrCLIP.

Produces npz files with EXACTLY the format of CorrCLIP's official
pre-generated region_masks (key 'instance_mask', int16 [H_orig, W_orig],
0 = unsegmented union, 1..M = region ids), so evaluation only needs
`model.instance_mask_path` pointed at the output dir (mask_generator stays
None in the config).

Run this in an env whose timm supports DINOv3 (timm >= 1.0.20), e.g. the env
used for ~/dino_rtp_seg — it does NOT need mmseg/mmcv. Example (server):

  PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=3 python gen_gram_masks.py \
      --images ~/CorrCLIP/data/ade/ADEChallengeData2016/images/validation \
      --out ~/CorrCLIP/data/region_masks_gram/ade/k64_hard_nearest_s896 \
      --weights ~/dino_rtp_seg/dinov3_vits16.safetensors --k 64

--variant also accepts dinov2_vits14 / dinov2_vitb14 (patch14, DINOv2, no
registers) as a backbone-attribution control -- same pipeline, older
backbone, isolates whether the gain is the method or just DINOv3. DINOv2
has no local weights in this repo: leave --weights unset (or pointing at a
nonexistent file) and it auto-downloads via timm pretrained=True (any HF
mirror should be set by the caller via HF_ENDPOINT, this script does not
touch it). Example (server, needs outbound network / HF mirror):

  PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=3 HF_ENDPOINT=https://hf-mirror.com \
      python gen_gram_masks.py \
      --images ~/CorrCLIP/data/ade/ADEChallengeData2016/images/validation \
      --out ~/CorrCLIP/data/region_masks_gram/ade/dinov2_vits14_k64_hard_nearest_s896 \
      --variant dinov2_vits14 --k 64 --multiscale --pamr-iters 10

Smoke test without weights / GPU (random init, synthetic image):
  python gen_gram_masks.py --smoke
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gram_mask_generator import GramMaskGenerator  # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", help="dir with validation images (ADE val: 2000 .jpg)")
    ap.add_argument("--out", help="output dir for <stem>.npz files")
    # NOTE: 上服务器后先 `ls ~/dino_rtp_seg/` 确认 safetensors 实际文件名
    # default is resolved after parsing (depends on --variant): DINOv3
    # variants default to the local safetensors below; DINOv2 variants have
    # none, so an unset/missing --weights falls back to timm's own
    # pretrained=True hub download (see --no-download).
    ap.add_argument("--weights", default=None,
        help="local safetensors (timm-converted). DINOv3 variants "
             "(vits16/vitb16) default to ~/dino_rtp_seg/dinov3_<variant>"
             ".safetensors and require a local file to exist. DINOv2 "
             "variants (dinov2_vits14/dinov2_vitb14) have no local weights "
             "-- leave unset (or point at a nonexistent file) to "
             "auto-download via timm; pass an existing local safetensors "
             "to use that instead.")
    ap.add_argument("--variant", default="vits16",
                    choices=["vits16", "vitb16", "dinov2_vits14", "dinov2_vitb14"],
                    help="vits16/vitb16: DINOv3, patch16, local weights. "
                         "dinov2_vits14/dinov2_vitb14: DINOv2, patch14, "
                         "backbone-attribution control (see --weights, "
                         "--no-download)")
    ap.add_argument("--no-download", action="store_true",
                    help="for DINOv2 variants with no usable local weights: "
                         "force a random-init backbone instead of timm's "
                         "pretrained=True hub download (offline smoke tests "
                         "use this automatically)")
    ap.add_argument("--k", type=int, default=64, help="number of FPS seeds (32/64/128)")
    ap.add_argument("--tau", type=float, default=0.07, help="affinity temperature")
    ap.add_argument("--refine-rounds", type=int, default=1,
                    help="extra rounds of region-mean re-propagation (proto default 1)")
    ap.add_argument("--binarize", default="hard", choices=["hard", "topp"],
                    help="hard: mutually exclusive argmax partition; "
                         "topp: per-token nucleus sets painted by ascending area")
    ap.add_argument("--top-p", type=float, default=0.9, help="nucleus mass for topp")
    ap.add_argument("--conf-thresh", type=float, default=0.0,
                    help=">0: tokens with max region prob below this become 0 "
                         "(unsegmented union). 0 = full partition")
    ap.add_argument("--img-size", type=int, default=896,
                    help="longest side fed to the backbone (snapped to a "
                         "multiple of the variant's patch size: 16 for "
                         "DINOv3, 14 for DINOv2)")
    ap.add_argument("--upsample", default="nearest", choices=["nearest", "soft"],
                    help="nearest: label map nearest-upsampled to original size; "
                         "soft: bilinear on probs/masks, decision at full res")
    # --- training-free refinements (default OFF = bit-identical baseline) ---
    ap.add_argument("--multiscale", action="store_true",
                    help="extra coarse pass; fine regions merged under coarse "
                         "regions by feature similarity (anti over-segmentation)")
    ap.add_argument("--coarse-size", type=int, default=0,
                    help="coarse pass longest side (0 = img_size/2)")
    ap.add_argument("--coarse-k", type=int, default=0,
                    help="FPS seeds for the coarse pass (0 = k/2, min 8)")
    ap.add_argument("--merge-thresh", type=float, default=0.6,
                    help="cos(fine region mean, coarse group mean) needed to "
                         "absorb a fine region into its coarse group")
    ap.add_argument("--pamr-iters", type=int, default=0,
                    help=">0 enables PAMR boundary refinement at pixel level "
                         "(10 = NACLIP setting)")
    ap.add_argument("--pamr-dilations", default="1,2,4,8,12,24",
                    help="comma-separated PAMR dilations")
    ap.add_argument("--pamr-res", type=int, default=0,
                    help="cap on PAMR resolution (longest side); 0 = native")
    ap.add_argument("--attn-mix", type=float, default=0.0,
                    help=">0: blend token cosine with last-layer key-key "
                         "cosine, sim = (1-a)*cos_f + a*cos_k (experimental)")
    # --- trivial-clustering controls (red-team baselines) --------------------
    ap.add_argument("--cluster", default="gram",
                    choices=["gram", "kmeans", "slic"],
                    help="gram: FPS seeds + one-step Gram propagation (ours, "
                         "default, bit-identical to before); kmeans: spherical "
                         "k-means on the same DINOv3 tokens (trivial-clustering "
                         "control); slic: skimage SLIC superpixels in RGB, no "
                         "features (pure-image control, --weights ignored)")
    ap.add_argument("--kmeans-iters", type=int, default=15,
                    help="Lloyd iterations for --cluster kmeans")
    ap.add_argument("--seed", type=int, default=0,
                    help="k-means random-init seed (fixed for reproducibility)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-fp16", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N images")
    ap.add_argument("--overwrite", action="store_true",
                    help="regenerate even if npz exists")
    ap.add_argument("--smoke", action="store_true",
                    help="CPU self-test with random weights, no data needed")
    return ap.parse_args()


def build_generator(args, weights_file):
    dil = tuple(int(d) for d in str(args.pamr_dilations).split(",") if d)
    return GramMaskGenerator(
        weights_file=weights_file, variant=args.variant, k=args.k, tau=args.tau,
        refine_rounds=args.refine_rounds, binarize=args.binarize, top_p=args.top_p,
        conf_thresh=args.conf_thresh, img_size=args.img_size,
        upsample=args.upsample, device=args.device, fp16=not args.no_fp16,
        multiscale=args.multiscale, coarse_size=args.coarse_size,
        coarse_k=args.coarse_k, merge_thresh=args.merge_thresh,
        pamr_iters=args.pamr_iters, pamr_dilations=dil, pamr_res=args.pamr_res,
        attn_mix=args.attn_mix, cluster=args.cluster,
        kmeans_iters=args.kmeans_iters, seed=args.seed,
        download=not getattr(args, "no_download", False))


# DINOv3 variants' default local weights path (server layout), keyed by
# --variant. DINOv2 variants have no entry here -> no default, empty
# --weights falls through to timm pretrained=True auto-download.
_DEFAULT_DINOV3_WEIGHTS = {
    "vits16": os.path.expanduser("~/dino_rtp_seg/dinov3_vits16.safetensors"),
    "vitb16": os.path.expanduser("~/dino_rtp_seg/dinov3_vitb16.safetensors"),
}


def _is_dinov2(variant):
    return variant.startswith("dinov2_")


def resolve_weights(args):
    """Fill in --weights default (variant-dependent) and validate it.

    DINOv3 variants: must resolve to an existing local file (unchanged
    behavior). DINOv2 variants: an existing local file is used if given;
    otherwise --weights is normalized to None, signalling the
    pretrained=True auto-download path in GramMaskGenerator/load_backbone.
    """
    if args.cluster == "slic":               # slic never touches the backbone
        return
    if _is_dinov2(args.variant):
        if args.weights and os.path.isfile(args.weights):
            return                            # use the given local file
        args.weights = None                   # -> timm pretrained=True (or
                                               #    random init if --no-download)
        return
    if args.weights is None:
        args.weights = _DEFAULT_DINOV3_WEIGHTS.get(args.variant, "")
    assert args.weights and os.path.isfile(args.weights), \
        f"weights not found: {args.weights!r} (ls ~/dino_rtp_seg/ to check the name)"


def smoke(args, tag=""):
    """Shape/dtype/semantics self-test on synthetic images, random weights."""
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    gen = build_generator(args, weights_file=None)
    rng = np.random.RandomState(0)
    # odd sizes on purpose: not divisible by 16; also a tiny image (K > N case)
    for (h, w) in [(497, 683), (300, 300), (37, 51)]:
        if getattr(args, "cluster", "gram") == "slic":
            # SLIC degenerates to 1 segment on iid noise; give it a low-freq
            # image (8x8 noise bilinearly upscaled) like natural statistics
            small = rng.randint(0, 255, (8, 8, 3), dtype=np.uint8)
            img = Image.fromarray(small).resize((w, h), Image.BILINEAR)
        else:
            img = Image.fromarray(rng.randint(0, 255, (h, w, 3), dtype=np.uint8))
        m = gen.generate(img)
        assert m.shape == (h, w), (m.shape, (h, w))
        assert m.dtype == torch.int32
        vals = torch.unique(m)
        assert vals.min() >= 0 and vals.max() < 10000
        if getattr(args, "cluster", "gram") == "slic":
            assert len(vals) > 1, "slic collapsed to a single segment"
        # ids must be compact: 0..M or 1..M with no gaps
        expect = torch.arange(int(vals.min()), int(vals.max()) + 1)
        assert torch.equal(vals.cpu(), expect), vals
        arr = m.cpu().numpy().astype(np.int16)
        buf = Path("/tmp") / "gram_smoke.npz"
        np.savez_compressed(buf, instance_mask=arr)
        back = np.load(buf)["instance_mask"]
        assert back.shape == (h, w) and back.dtype == np.int16
        print(f"  smoke ok {h}x{w}: {len(vals)} unique ids, "
              f"min={int(vals.min())}, max={int(vals.max())}")
    print(f"SMOKE PASSED (binarize={args.binarize}, upsample={args.upsample}"
          f"{', ' + tag if tag else ''})")


def smoke_pamr_unit():
    """PAMR convergence + distribution-preservation check on a synthetic
    two-color image with a noisy soft mask."""
    from gram_mask_generator import pamr
    torch.manual_seed(0)
    h, w = 96, 128
    img = torch.zeros(1, 3, h, w)
    img[:, 0, :, : w // 2] = 1.0                 # red left, black right
    logits = torch.randn(1, 4, h, w) * 0.5
    logits[0, 0, :, : w // 2] += 2.0
    logits[0, 1, :, w // 2:] += 2.0
    probs = torch.softmax(logits, dim=1)
    deltas, prev = [], probs
    for _ in range(3):                           # 3x5 iters, track step size
        cur = pamr(img, prev, iters=5)
        s = cur.sum(1)
        assert torch.isfinite(cur).all()
        assert (s - 1).abs().max() < 1e-3, "channel sums must stay ~1"
        deltas.append((cur - prev).abs().mean().item())
        prev = cur
    assert deltas[-1] < deltas[0], f"PAMR not converging: {deltas}"
    # refined argmax must respect the color edge better than the noisy input
    err_in = (probs.argmax(1)[0, :, : w // 2] != 0).float().mean()
    err_out = (prev.argmax(1)[0, :, : w // 2] != 0).float().mean()
    assert err_out <= err_in, (err_in, err_out)
    print(f"  pamr unit ok: step deltas {['%.4f' % d for d in deltas]}, "
          f"left-half argmax err {err_in:.3f} -> {err_out:.3f}")


def smoke_dinov2_tokens():
    """DINOv2 backbone-attribution control: direct token-grid + prefix-token
    unit check (patch14, no registers -> num_prefix_tokens=1), offline
    (download=False -> random-init architecture, no network/hub access).

    Confirms grid == (H/14, W/14) (not the DINOv3 /16 assumption) and that
    the cls/register-stripping in GramMaskGenerator._tokens reads
    self.num_prefix from the model rather than a hardcoded constant."""
    from gram_mask_generator import GramMaskGenerator
    gen = GramMaskGenerator(weights_file=None, variant="dinov2_vits14",
                            device="cpu", fp16=False, img_size=98,
                            download=False)
    assert gen.patch == 14, gen.patch
    assert gen.num_prefix == 1, (
        f"DINOv2 should have num_prefix_tokens=1 (no registers), got "
        f"{gen.num_prefix}")
    img = Image.fromarray(
        np.random.RandomState(0).randint(0, 255, (70, 98, 3), dtype=np.uint8))
    x, grid = gen._preprocess(img)
    assert grid == (5, 7), grid                     # 70/14=5, 98/14=7
    feat = gen._tokens(x, grid)
    assert feat.shape[0] == grid[0] * grid[1] == 35, feat.shape
    print(f"  dinov2 tokens unit ok: patch={gen.patch}, "
          f"num_prefix={gen.num_prefix}, grid={grid}, tokens={feat.shape[0]}")


def smoke_kmeans_unit():
    """Spherical k-means: converges within 15 Lloyd rounds on separable
    data, is deterministic under a fixed seed, and recovers the planted
    clusters."""
    import torch.nn.functional as F
    from gram_mask_generator import spherical_kmeans_probs
    g = torch.Generator().manual_seed(0)
    centers = F.normalize(torch.randn(4, 32, generator=g), dim=-1)
    feat = centers.repeat_interleave(50, 0) + 0.03 * torch.randn(200, 32, generator=g)
    p15 = spherical_kmeans_probs(feat, k=8, iters=15, seed=0)
    p50 = spherical_kmeans_probs(feat, k=8, iters=50, seed=0)
    assert torch.equal(p15, p50), "kmeans did not reach a fixpoint in 15 iters"
    assert torch.equal(p15, spherical_kmeans_probs(feat, k=8, iters=15, seed=0)), \
        "kmeans not deterministic under fixed seed"
    assert p15.shape[1] == 200 and torch.all(p15.sum(0) == 1)  # one-hot columns
    labels = p15.argmax(0)
    planted = torch.arange(4).repeat_interleave(50)
    # k=8 > 4 planted clusters: splitting a planted cluster is fine, but no
    # kmeans cluster may MIX planted clusters (majority-fraction purity)
    pure = sum(planted[labels == c].bincount().max().item()
               for c in labels.unique().tolist()) / 200.0
    assert pure > 0.99, f"kmeans mixed the planted clusters: purity={pure}"
    assert spherical_kmeans_probs(feat[:5], k=8).shape[0] <= 5  # K > N case
    print(f"  kmeans unit ok: 15-iter fixpoint, deterministic, purity={pure:.3f}")


def run_all_smokes(args):
    for b in ("hard", "topp"):
        for u in ("nearest", "soft"):
            args.binarize, args.upsample = b, u
            args.k = 64
            smoke(args)
    smoke_pamr_unit()
    # refinement switches (on the mainline hard/nearest config)
    args.binarize, args.upsample = "hard", "nearest"
    for name, patch in [
            ("multiscale", dict(multiscale=True)),
            ("pamr", dict(pamr_iters=3)),
            ("both", dict(multiscale=True, pamr_iters=3)),
            ("pamr_capped", dict(pamr_iters=3, pamr_res=256)),
            ("attn_mix", dict(attn_mix=0.3)),
            ("all", dict(multiscale=True, pamr_iters=3, attn_mix=0.3))]:
        base = dict(multiscale=False, pamr_iters=0, pamr_res=0, attn_mix=0.0)
        for k, v in {**base, **patch}.items():
            setattr(args, k, v)
        smoke(args, tag=name)
    # trivial-clustering baselines: same output contract in all three modes
    for k, v in dict(multiscale=False, pamr_iters=0, pamr_res=0,
                     attn_mix=0.0).items():
        setattr(args, k, v)
    smoke_kmeans_unit()
    args.cluster = "kmeans"
    smoke(args, tag="cluster=kmeans")
    args.cluster = "slic"
    try:
        smoke(args, tag="cluster=slic")
    except ImportError as e:
        print(f"  SKIP slic smoke: {e}")
    args.cluster = "gram"

    # DINOv2 backbone-attribution control (patch14, no registers). Token
    # grid + prefix-stripping unit check, then the full generate() contract
    # (baseline / multiscale / PAMR / both) -- all offline via --no-download
    # (random-init architecture, no hub access needed).
    smoke_dinov2_tokens()
    saved_variant, saved_no_dl = args.variant, getattr(args, "no_download", False)
    args.variant, args.no_download = "dinov2_vits14", True
    for name, patch in [
            ("dinov2_baseline", dict(multiscale=False, pamr_iters=0)),
            ("dinov2_multiscale", dict(multiscale=True, pamr_iters=0)),
            ("dinov2_pamr", dict(multiscale=False, pamr_iters=3)),
            ("dinov2_both", dict(multiscale=True, pamr_iters=3))]:
        base = dict(multiscale=False, pamr_iters=0, pamr_res=0, attn_mix=0.0)
        for k, v in {**base, **patch}.items():
            setattr(args, k, v)
        smoke(args, tag=name)
    args.variant, args.no_download = saved_variant, saved_no_dl
    print("ALL SMOKES PASSED")


def main():
    args = parse_args()
    if args.smoke:
        run_all_smokes(args)
        return

    assert args.images and args.out, "--images and --out are required"
    resolve_weights(args)                    # fills default / validates --weights
    os.makedirs(args.out, exist_ok=True)
    # record exact settings next to the masks (official script does the same)
    with open(os.path.join(args.out, "config.txt"), "w") as f:
        json.dump(vars(args), f, indent=2)

    files = sorted(p for p in Path(args.images).iterdir()
                   if p.suffix.lower() in IMG_EXTS)
    assert files, f"no images under {args.images}"
    if args.limit:
        files = files[:args.limit]

    gen = build_generator(args, weights_file=args.weights)
    t0, n_regions, done, skipped = time.time(), 0, 0, 0
    for i, p in enumerate(files):
        out_path = Path(args.out) / (p.stem + ".npz")
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue
        img = Image.open(p).convert("RGB")
        mask = gen.generate(img).cpu().numpy().astype(np.int16)
        np.savez_compressed(out_path, instance_mask=mask)
        n_regions += int(mask.max())
        done += 1
        if done % 100 == 0:
            dt = time.time() - t0
            print(f"[{i + 1}/{len(files)}] {dt / done:.2f}s/img, "
                  f"avg {n_regions / done:.1f} regions/img", flush=True)
    dt = time.time() - t0
    print(f"DONE: {done} generated, {skipped} skipped, "
          f"{dt / max(done, 1):.2f}s/img, "
          f"avg {n_regions / max(done, 1):.1f} regions/img -> {args.out}")


if __name__ == "__main__":
    main()
