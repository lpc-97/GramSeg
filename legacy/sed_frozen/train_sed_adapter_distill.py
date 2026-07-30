"""Step-2 distillation variant: train the adapter f: frozen_feat -> official_feat.

Instead of the CE proxy loss (which co-adapts against the decoder and shifts
the feature distribution, breaking the cost volume: 31.74->24.17 mIoU), we
directly regress the frozen (LAION-visual) clip_vis_dense features onto the
official fine-tuned-visual features on the same images:

  loss = w_cos * (1 - cos(f(x), y)) + w_l2 * relMSE(f(x), y)

  relMSE = mean_tokens[ mean_c (f(x)-y)^2 / (mean_c y^2 + eps) ]
  (scale-free, so w_l2 is comparable to the cosine term regardless of the
   absolute feature magnitude; cosine is the operative term since
   Aggregator.correlation renormalizes img_feats anyway.)

Pairing: the two caches were built by cache_sed_features.py over the same
sorted 20k COCO-Stuff train list with the same shard size, so shard basenames
and the per-shard "files" lists must match exactly — both are asserted.

No GT needed: all 576 tokens per image are used (soft labels are dropped at
load time to halve RAM; ~15GB per cache for 20k images, ~30GB total).

Held-out eval: mean token cosine to the official features BEFORE (identity,
i.e. raw frozen vs official) and AFTER the adapter, plus relMSE.

The adapter (adapter_module.Adapter) zero-inits its last linear, so at step 0
f(x) = x exactly: training starts from the LAION features and the initial val
cosine equals the frozen-vs-official baseline. Checkpoint format matches
train_sed_adapter.py, so Adapter.from_checkpoint / sed_adapter_inject work
unchanged. Pure optimization wall-clock is printed (the cost-story number).
"""
import argparse
import glob
import os
import time

import torch
import torch.nn.functional as F

from adapter_module import Adapter


def load_paired_caches(frozen_dir: str, official_dir: str, max_images: int):
    """Load feats from both caches, strictly pair-checked (shard names + files).

    Returns (frozen_feats, official_feats) fp16 CPU tensors of identical shape.
    """
    fz = sorted(glob.glob(os.path.join(frozen_dir, "shard_*.pt")))
    of = sorted(glob.glob(os.path.join(official_dir, "shard_*.pt")))
    assert fz, f"no shards in {frozen_dir}"
    assert of, f"no shards in {official_dir}"
    fz_names = [os.path.basename(p) for p in fz]
    of_names = [os.path.basename(p) for p in of]
    assert fz_names == of_names, (
        f"shard sets differ: {len(fz_names)} vs {len(of_names)} "
        f"(first mismatch: "
        f"{next((a, b) for a, b in zip(fz_names, of_names) if a != b) if len(fz_names) == len(of_names) else (fz_names[:3], of_names[:3])})")

    d0 = torch.load(fz[0], map_location="cpu")
    shard_size, C, g, _ = d0["feats"].shape
    n_max = shard_size * len(fz)
    if max_images:
        n_max = min(n_max, max_images)
    x = torch.empty((n_max, C, g, g), dtype=torch.float16)  # frozen
    y = torch.empty((n_max, C, g, g), dtype=torch.float16)  # official
    n = 0
    for i, (pf, po) in enumerate(zip(fz, of)):
        df = d0 if i == 0 else torch.load(pf, map_location="cpu")
        do = torch.load(po, map_location="cpu")
        # strict same-image same-order pairing inside the shard
        assert df["files"] == do["files"], \
            f"file-order mismatch in {os.path.basename(pf)}"
        assert df["feats"].shape == do["feats"].shape
        b = min(df["feats"].shape[0], n_max - n)
        x[n:n + b] = df["feats"][:b]
        y[n:n + b] = do["feats"][:b]
        n += b
        del df, do  # drop softs/feats of this shard before loading the next
        if n >= n_max:
            break
    return x[:n], y[:n]


def batch_tokens(feats, idx, dev):
    """(b,C,g,g) fp16 CPU -> (b*g*g, C) fp32 on dev."""
    f = feats[idx].to(dev).float()                     # fp16 -> fp32 on GPU
    return f.flatten(2).transpose(1, 2).reshape(-1, f.shape[1])


def distill_terms(pred, target, eps=1e-6):
    """Returns (1-cos).mean(), relMSE.mean() over tokens; all fp32."""
    cos = F.cosine_similarity(pred, target, dim=-1)
    rel = ((pred - target) ** 2).mean(-1) / (target.pow(2).mean(-1) + eps)
    return (1.0 - cos).mean(), rel.mean()


@torch.no_grad()
def evaluate(net, x, y, dev, bs=64, identity=False):
    """Mean token cosine + relMSE of (adapter|identity)(frozen) vs official."""
    cos_sum = rel_sum = tok = 0.0
    for i in range(0, x.shape[0], bs):
        idx = torch.arange(i, min(i + bs, x.shape[0]))
        e = batch_tokens(x, idx, dev)
        t = batch_tokens(y, idx, dev)
        z = e if identity else net(e)
        cos = F.cosine_similarity(z, t, dim=-1)
        rel = ((z - t) ** 2).mean(-1) / (t.pow(2).mean(-1) + 1e-6)
        cos_sum += cos.sum().item()
        rel_sum += rel.sum().item()
        tok += cos.numel()
    return cos_sum / max(tok, 1), rel_sum / max(tok, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen", required=True,
                    help="cache dir of the frozen/LAION features (input)")
    ap.add_argument("--official", required=True,
                    help="cache dir of the official fine-tuned features (target)")
    ap.add_argument("--out", default=None,
                    help="default: <frozen>/adapter_distill.pth")
    ap.add_argument("--hidden", type=int, default=3072,
                    help="MLP width (~3.94M params @3072)")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--bs", type=int, default=32, help="images per step (x576 tokens)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--w-cos", type=float, default=1.0, help="weight of 1-cosine term")
    ap.add_argument("--w-l2", type=float, default=0.5, help="weight of relative-MSE term")
    ap.add_argument("--max-images", type=int, default=0, help="0 = all cached")
    ap.add_argument("--val-frac", type=float, default=0.02)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out_path = args.out or os.path.join(args.frozen, "adapter_distill.pth")

    t_load = time.time()
    x, y = load_paired_caches(args.frozen, args.official, args.max_images)
    n_val = max(int(x.shape[0] * args.val_frac), 1)
    tr_x, tr_y = x[:-n_val], y[:-n_val]
    va_x, va_y = x[-n_val:], y[-n_val:]
    d = x.shape[1]
    print(f"[distill] paired cache {tuple(x.shape)} d={d} "
          f"train {tr_x.shape[0]} val {n_val} (load {time.time() - t_load:.0f}s)")

    net = Adapter(d=d, hidden=args.hidden).to(dev)
    n_par = sum(p.numel() for p in net.parameters())
    print(f"[distill] adapter params: {n_par / 1e6:.2f}M (hidden={args.hidden}) "
          f"loss = {args.w_cos}*(1-cos) + {args.w_l2}*relMSE")

    cos0, rel0 = evaluate(net, va_x, va_y, dev, identity=True)
    print(f"[distill] val BEFORE (raw frozen vs official): "
          f"cos {cos0:.4f} relMSE {rel0:.4f}")

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
    n = tr_x.shape[0]
    steps = args.epochs * (n // args.bs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)

    t0 = time.time()                        # pure optimization wall-clock
    step = 0
    for ep in range(args.epochs):
        perm = torch.randperm(n)
        for i in range(0, n - args.bs + 1, args.bs):
            idx = perm[i:i + args.bs]
            e = batch_tokens(tr_x, idx, dev)
            t = batch_tokens(tr_y, idx, dev)
            l_cos, l_rel = distill_terms(net(e), t)
            loss = args.w_cos * l_cos + args.w_l2 * l_rel
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            if step % 100 == 0 or step == steps:
                print(f"ep{ep} step {step}/{steps} loss {loss.item():.4f} "
                      f"(1-cos {l_cos.item():.4f} relMSE {l_rel.item():.4f}) "
                      f"gate {net.gate.item():.3f} ({time.time() - t0:.0f}s)",
                      flush=True)
    train_secs = time.time() - t0

    cos1, rel1 = evaluate(net, va_x, va_y, dev)
    print(f"[distill] val AFTER: cos {cos1:.4f} relMSE {rel1:.4f} "
          f"(before cos {cos0:.4f} relMSE {rel0:.4f})")
    torch.save({"state_dict": net.state_dict(), "d": d, "hidden": args.hidden,
                "train_seconds": train_secs,
                "val_cos": cos1, "val_cos_raw": cos0,
                "val_relmse": rel1, "val_relmse_raw": rel0,
                "w_cos": args.w_cos, "w_l2": args.w_l2,
                "frozen_cache": os.path.abspath(args.frozen),
                "official_cache": os.path.abspath(args.official)},
               out_path)
    print(f"ADAPTER_DONE train_wall_clock={train_secs:.1f}s "
          f"params={n_par / 1e6:.2f}M val_cos {cos0:.4f}->{cos1:.4f} -> {out_path}")


if __name__ == "__main__":
    main()
