"""Train the Step-2 adapter on cached SED cost-map features (seconds~minutes).

Objective: per-token 171-class CE between the adapted feature and the COCO
text embeddings, folded over templates exactly the way SED folds them for
text guidance: normalize per template -> mean over the 80 templates ->
renormalize (Aggregator.feature_map / text_guidance semantics; correlation
itself keeps templates as channels, the folded matrix is the natural single
classifier for a pointwise loss).

logits = normalize(adapter(e)) @ text_folded.T * exp(logit_scale)

Tokens whose 32x32 GT block is more than half ignore-label are dropped;
remaining soft labels are renormalized to sum 1.

Reports pure optimization wall-clock (excl. cache load) — this is the number
for the cost story.
"""
import argparse
import glob
import os
import time

import torch
import torch.nn.functional as F

from adapter_module import Adapter


def load_cache(cache_dir: str, max_images: int):
    """Preallocated load (peak RAM = cache size + one shard, ~15-19GB @20k)."""
    shards = sorted(glob.glob(os.path.join(cache_dir, "shard_*.pt")))
    assert shards, f"no shards in {cache_dir}"
    first = torch.load(shards[0], map_location="cpu")
    f0, s0 = first["feats"], first["softs"]
    shard_size = f0.shape[0]
    # upper bound on N; trimmed after the last shard is read
    n_max = shard_size * len(shards)
    if max_images:
        n_max = min(n_max, max_images)
    feats = torch.empty((n_max,) + tuple(f0.shape[1:]), dtype=torch.float16)
    softs = torch.empty((n_max,) + tuple(s0.shape[1:]), dtype=torch.float16)
    n = 0
    for i, sh in enumerate(shards):
        d = first if i == 0 else torch.load(sh, map_location="cpu")
        b = min(d["feats"].shape[0], n_max - n)
        feats[n:n + b] = d["feats"][:b]
        softs[n:n + b] = d["softs"][:b]
        n += b
        if n >= n_max:
            break
    return feats[:n], softs[:n]


def fold_text(text_path: str, dev: str):
    t = torch.load(text_path, map_location="cpu")["text"].float()   # (T,P,C)
    t = F.normalize(t, dim=-1).mean(dim=1)                          # fold P
    t = F.normalize(t, dim=-1)                                      # (T,C)
    return t.to(dev)


def tokens_of(feats, softs, idx, dev):
    """(b,C,g,g)+(b,T,g,g) -> valid (M,C) float feats, (M,T) renormalized soft."""
    e = feats[idx].to(dev).float().flatten(2).transpose(1, 2).reshape(-1, feats.shape[1])
    s = softs[idx].to(dev).float().flatten(2).transpose(1, 2).reshape(-1, softs.shape[1])
    cov = s.sum(-1)
    valid = cov > 0.5                       # >=50% of the block is labeled
    e, s = e[valid], s[valid] / cov[valid, None]
    return e, s


@torch.no_grad()
def token_acc(net, feats, softs, text, dev, bs=64, identity=False):
    correct = total = 0
    for i in range(0, feats.shape[0], bs):
        idx = torch.arange(i, min(i + bs, feats.shape[0]))
        e, s = tokens_of(feats, softs, idx, dev)
        if e.numel() == 0:
            continue
        z = F.normalize(e if identity else net(e), dim=-1)
        pred = (z @ text.t()).argmax(-1)
        correct += (pred == s.argmax(-1)).sum().item()
        total += pred.numel()
    return correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="cache dir from cache_sed_features.py")
    ap.add_argument("--text", default=None, help="default: <cache>/text_features.pt")
    ap.add_argument("--out", default=None, help="default: <cache>/adapter.pth")
    ap.add_argument("--hidden", type=int, default=3072, help="MLP width (~4M params @3072)")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--bs", type=int, default=32, help="images per step (x576 tokens)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--max-images", type=int, default=0, help="0 = all cached")
    ap.add_argument("--val-frac", type=float, default=0.02)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    text_path = args.text or os.path.join(args.cache, "text_features.pt")
    out_path = args.out or os.path.join(args.cache, "adapter.pth")

    t_load = time.time()
    feats, softs = load_cache(args.cache, args.max_images)
    text = fold_text(text_path, dev)
    n_val = max(int(feats.shape[0] * args.val_frac), 1)
    tr_f, tr_s = feats[:-n_val], softs[:-n_val]
    va_f, va_s = feats[-n_val:], softs[-n_val:]
    d = feats.shape[1]
    print(f"[train] cache {tuple(feats.shape)} d={d} text {tuple(text.shape)} "
          f"train {tr_f.shape[0]} val {n_val} (load {time.time() - t_load:.0f}s)")

    net = Adapter(d=d, hidden=args.hidden).to(dev)
    n_par = sum(p.numel() for p in net.parameters())
    print(f"[train] adapter params: {n_par / 1e6:.2f}M (hidden={args.hidden})")

    acc0 = token_acc(net, va_f, va_s, text, dev, identity=True)
    print(f"[train] val token-acc BEFORE (raw features): {acc0:.4f}")

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
    n = tr_f.shape[0]
    steps = args.epochs * (n // args.bs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)

    t0 = time.time()                       # pure optimization wall-clock
    step = 0
    for ep in range(args.epochs):
        perm = torch.randperm(n)
        for i in range(0, n - args.bs + 1, args.bs):
            e, s = tokens_of(tr_f, tr_s, perm[i:i + args.bs], dev)
            if e.numel() == 0:
                continue
            logits = F.normalize(net(e), dim=-1) @ text.t()
            logits = logits * net.logit_scale.exp().clamp(max=100)
            # soft-target CE, explicit (works on any torch version)
            loss = -(s * F.log_softmax(logits, dim=-1)).sum(-1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            if step % 100 == 0 or step == steps:
                acc = (logits.argmax(-1) == s.argmax(-1)).float().mean().item()
                print(f"ep{ep} step {step}/{steps} loss {loss.item():.3f} "
                      f"acc {acc:.3f} gate {net.gate.item():.3f} "
                      f"({time.time() - t0:.0f}s)", flush=True)
    train_secs = time.time() - t0

    acc1 = token_acc(net, va_f, va_s, text, dev)
    print(f"[train] val token-acc AFTER: {acc1:.4f} (before {acc0:.4f})")
    torch.save({"state_dict": net.state_dict(), "d": d, "hidden": args.hidden,
                "train_seconds": train_secs, "val_acc": acc1,
                "val_acc_raw": acc0, "cache": os.path.abspath(args.cache)},
               out_path)
    print(f"ADAPTER_DONE train_wall_clock={train_secs:.1f}s "
          f"params={n_par / 1e6:.2f}M -> {out_path}")


if __name__ == "__main__":
    main()
