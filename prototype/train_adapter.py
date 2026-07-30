"""Train a residual adapter on cached SigLIP region embeddings (minutes).

adapter(emb) = emb + MLP(emb); classify vs SigLIP text embeddings with CE
against cached soft GT labels. Saves adapter.pt.
"""
import argparse, os, time
import torch
import torch.nn as nn
import torch.nn.functional as F


class Adapter(nn.Module):
    def __init__(self, d=1152, hidden=768):
        super().__init__()
        self.mlp = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, hidden),
                                 nn.GELU(), nn.Linear(hidden, d))
        self.gate = nn.Parameter(torch.tensor(0.1))
        self.logit_scale = nn.Parameter(torch.tensor(4.0))

    def forward(self, e):
        return F.normalize(e + self.gate * self.mlp(e), dim=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache_regions")
    ap.add_argument("--text", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=256, help="images per batch")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default="adapter.pt")
    args = ap.parse_args()
    dev = "cuda"

    text = F.normalize(torch.load(args.text)["embeddings"], dim=-1).to(dev)
    files = sorted(os.listdir(args.cache))
    print(f"cached images: {len(files)}")
    # load everything into RAM (20k x 128 x 1302 fp16 ~ 6.7GB)
    embs, softs = [], []
    for f in files:
        d = torch.load(os.path.join(args.cache, f))
        embs.append(d["emb"]); softs.append(d["soft"])
    embs = torch.stack(embs); softs = torch.stack(softs)   # (I,K,1152),(I,K,150)
    print("loaded", embs.shape)

    net = Adapter().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.01)
    n = len(files)
    steps = args.epochs * (n // args.bs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    t0 = time.time()
    step = 0
    for ep in range(args.epochs):
        perm = torch.randperm(n)
        for i in range(0, n - args.bs + 1, args.bs):
            idx = perm[i:i + args.bs]
            e = embs[idx].to(dev).float().flatten(0, 1)     # (B*K,1152)
            s = softs[idx].to(dev).float().flatten(0, 1)    # (B*K,150)
            valid = s.sum(-1) > 0.5                          # labeled regions
            e, s = e[valid], s[valid]
            logits = net(e) @ text.t() * net.logit_scale.exp().clamp(max=100)
            loss = F.cross_entropy(logits, s)
            opt.zero_grad(set_to_none=True)
            loss.backward(); opt.step(); sched.step()
            step += 1
            if step % 200 == 0:
                acc = (logits.argmax(-1) == s.argmax(-1)).float().mean()
                print(f"ep{ep} step{step}/{steps} loss {loss.item():.3f} "
                      f"region-acc {acc.item():.3f}", flush=True)
    torch.save(net.state_dict(), args.out)
    print(f"ADAPTER_DONE {time.time()-t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
