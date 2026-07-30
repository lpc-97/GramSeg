"""Train DINO-RTP-Seg heads on ADE20K semantic segmentation.

Frozen: DINOv3 backbone + SigLIP text embeddings (precomputed).
Trained (~4M params): GramSeedProposal.TinyFormer, PromptEncoder, SparseRouter,
RefineHead, logit_scale.

Losses per image:
  L_mask     = dice + focal on composed refined masks of PRESENT classes
  L_align    = InfoNCE(routed embedding z_c -> class c among all 150)
  L_presence = BCE (present classes = 1, sampled absent = 0)

Usage:
  python train_ade.py --ade data/ADEChallengeData2016 --weights dinov3_vits16.safetensors \
      --text ade150_text.pt --iters 40000 --batch 8 --out ckpt
"""
import argparse, os, random, time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from dino_rtp_seg import DinoRTPSeg

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class ADESemSeg(Dataset):
    def __init__(self, root, split="training", size=640, native_gt=False):
        self.img_dir = os.path.join(root, "images", split)
        self.ann_dir = os.path.join(root, "annotations", split)
        self.files = sorted(os.listdir(self.img_dir))
        self.size = size
        self.train = split == "training"
        self.native_gt = native_gt                     # eval: keep GT at native res

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        fn = self.files[i]
        img = Image.open(os.path.join(self.img_dir, fn)).convert("RGB")
        ann = Image.open(os.path.join(self.ann_dir, fn.replace(".jpg", ".png")))
        s = self.size
        if self.train:                                  # random resized crop + flip
            w, h = img.size
            scale = random.uniform(0.75, 1.5) * s / min(w, h)
            nw, nh = max(s, int(w * scale)), max(s, int(h * scale))
            img = img.resize((nw, nh), Image.BILINEAR)
            ann = ann.resize((nw, nh), Image.NEAREST)
            x0, y0 = random.randint(0, nw - s), random.randint(0, nh - s)
            img = img.crop((x0, y0, x0 + s, y0 + s))
            ann = ann.crop((x0, y0, x0 + s, y0 + s))
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                ann = ann.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            img = img.resize((s, s), Image.BILINEAR)
            if not self.native_gt:
                ann = ann.resize((s, s), Image.NEAREST)
        x = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        x = (x - MEAN) / STD
        y = torch.from_numpy(np.array(ann)).long()      # 0=ignore, 1..150
        return x, y


def dice_loss(logits, target):
    p = torch.sigmoid(logits).flatten(1)
    t = target.flatten(1)
    num = 2 * (p * t).sum(-1) + 1.0
    den = p.sum(-1) + t.sum(-1) + 1.0
    return (1 - num / den).mean()


def focal_loss(logits, target, alpha=0.25, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p = torch.sigmoid(logits)
    pt = p * target + (1 - p) * (1 - target)
    w = (alpha * target + (1 - alpha) * (1 - target)) * (1 - pt) ** gamma
    return (w * bce).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ade", required=True)
    ap.add_argument("--coco", default=None, help="CocoStuff10k dir; overrides --ade for training")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--variant", default="vits16")
    ap.add_argument("--k", type=int, default=64, help="Gram-Seed regions at train time")
    ap.add_argument("--m", type=int, default=8, help="router top-m regions per prompt")
    ap.add_argument("--kernel", action="store_true", help="enable dynamic mask kernel")
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--iters", type=int, default=40000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--neg", type=int, default=16, help="absent classes sampled for presence loss")
    ap.add_argument("--out", default="ckpt")
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    dev = "cuda"
    tdata = torch.load(args.text)
    text_all = F.normalize(tdata["embeddings"], dim=-1).to(dev)
    NC = text_all.shape[0]                                # 150 (ADE) or 171 (COCO)

    model = DinoRTPSeg(args.variant, args.weights, img_size=args.size, k=args.k, m=args.m, use_kernel=args.kernel).to(dev)
    model.backbone.eval()
    decay, no_decay = [], []
    for nme, p in model.named_parameters():
        if nme.startswith("backbone.") or not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 else decay).append(p)   # norms/biases/logit_scale
    params = decay + no_decay
    print(f"trainable: {sum(p.numel() for p in params)/1e6:.2f}M")
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.05},
                             {"params": no_decay, "weight_decay": 0.0}], lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.iters)
    scaler = torch.amp.GradScaler("cuda")

    start_it = 0
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ck["model"], strict=False)
        opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        start_it = ck["it"] + 1
        print(f"resumed from {args.resume} at iter {start_it}")

    if args.coco:
        from coco_dataset import CocoStuff10k
        ds = CocoStuff10k(args.coco, args.size)
    else:
        ds = ADESemSeg(args.ade, "training", args.size)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=8,
                    pin_memory=True, drop_last=True, persistent_workers=True)
    print(f"train images: {len(ds)}")

    it, t0, run = start_it, time.time(), {}
    while it < args.iters:
        for x, y in dl:
            if it >= args.iters:
                break
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            B = x.shape[0]
            with torch.no_grad(), torch.amp.autocast("cuda", torch.bfloat16):
                feats, tapss, grid = model.backbone(x)   # frozen, batched (bf16 = eval regime)

            loss_mask = loss_align = loss_pres = x.new_zeros(())
            n_pres = 0
            with torch.amp.autocast("cuda", torch.bfloat16):
                masks_all, regions_all = model.proposal(feats)   # batched FPS
                for b in range(B):                       # per-image prompt routing
                    taps = [t[b:b+1] for t in tapss]
                    masks, regions = masks_all[b:b+1], regions_all[b:b+1]
                    present = torch.unique(y[b]); present = present[present > 0] - 1
                    if len(present) == 0:
                        continue
                    absent_pool = torch.tensor(
                        [c for c in range(NC) if c not in set(present.tolist())],
                        device=dev)
                    absent = absent_pool[torch.randperm(len(absent_pool))[:args.neg]]
                    cls = torch.cat([present, absent])
                    prompts = model.prompt_enc.encode_text(text_all[cls])
                    ymask, z, pres, w = model.router(prompts, regions, masks)

                    reg_hi, guide = model.refine(masks[0], taps, grid)   # (K,160,160)
                    # compose ALL routed classes; absent classes get all-zero
                    # targets so the eval-time 150-way argmax is actually trained
                    mask_hi = model.compose_hi(w, z, reg_hi, guide)       # present+absent

                    y4 = F.interpolate(y[b][None, None].float(),
                                       size=mask_hi.shape[-2:], mode="nearest")[0, 0]
                    gt = torch.stack([(y4 == c + 1).float() for c in cls])
                    loss_mask = loss_mask + dice_loss(mask_hi, gt) + focal_loss(mask_hi, gt)

                    zt = F.normalize(model.router.to_text(z[:len(present)]), dim=-1)
                    logits = zt @ text_all.t() * model.logit_scale.exp()
                    loss_align = loss_align + F.cross_entropy(logits, present)

                    # v2: region-level classification — every region gets a soft
                    # class target from its mask/GT overlap, so tail classes
                    # receive gradients on every image
                    rt = F.normalize(model.router.to_text(regions[0]), dim=-1)
                    rlogits = rt @ text_all.t() * model.logit_scale.exp()   # (K,150)
                    yp = F.interpolate(y[b][None, None].float(), size=grid,
                                       mode="nearest")[0, 0].long().flatten()
                    onehot = F.one_hot(yp, NC + 1).float()                  # 0=ignore
                    soft = masks[0] @ onehot[:, 1:]                         # (K,150)
                    denom = soft.sum(-1)
                    vr = denom > 1e-4                                       # regions w/ labeled pixels
                    if vr.any():
                        loss_align = loss_align + F.cross_entropy(
                            rlogits[vr], soft[vr] / denom[vr, None])

                    tgt = torch.cat([torch.ones(len(present)), torch.zeros(len(absent))]).to(dev)
                    loss_pres = loss_pres + F.binary_cross_entropy_with_logits(pres, tgt)
                    n_pres += 1

                if n_pres == 0:
                    continue
                loss = (2.0 * loss_mask + 1.0 * loss_align + 0.5 * loss_pres) / n_pres

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            with torch.no_grad():
                model.logit_scale.clamp_(max=4.6)        # ~log(100), CLIP-style

            for k, v in [("loss", loss), ("mask", loss_mask / n_pres),
                         ("align", loss_align / n_pres), ("pres", loss_pres / n_pres)]:
                run[k] = 0.98 * run.get(k, float(v)) + 0.02 * float(v)
            if it % 100 == 0:
                lr = sched.get_last_lr()[0]
                speed = 100 * args.batch / (time.time() - t0) if it > start_it else 0
                print(f"it {it:6d}/{args.iters} loss {run['loss']:.3f} "
                      f"(mask {run['mask']:.3f} align {run['align']:.3f} "
                      f"pres {run['pres']:.3f}) lr {lr:.2e} {speed:.1f} img/s",
                      flush=True)
                t0 = time.time()
            if it % 2000 == 0 and it > start_it:
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                            "sched": sched.state_dict(), "it": it},
                           f"{args.out}/latest.pt")
            it += 1

    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "sched": sched.state_dict(), "it": it}, f"{args.out}/final.pt")
    print("training done ->", f"{args.out}/final.pt")


if __name__ == "__main__":
    main()
