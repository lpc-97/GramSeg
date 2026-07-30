"""Evaluate DINO-RTP-Seg on ADE20K val (150-class semantic mIoU).

Protocol: resize to 640x640, per-class routed masks + alignment logits,
per-pixel argmax, IoU accumulated at 640 resolution against resized GT
(nearest). Reports mIoU, pixel acc, and FPS.

Usage: python eval_ade.py --ade data/ADEChallengeData2016 \
    --weights dinov3_vits16.safetensors --text ade150_text.pt --ckpt ckpt/latest.pt
"""
import argparse, os, time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from dino_rtp_seg import DinoRTPSeg
from train_ade import ADESemSeg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ade", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--variant", default="vits16")
    ap.add_argument("--k", type=int, default=64, help="number of Gram-Seed regions")
    ap.add_argument("--m", type=int, default=8, help="router top-m regions per prompt")
    ap.add_argument("--kernel", action="store_true", help="enable dynamic mask kernel")
    ap.add_argument("--readout", default="standard", choices=["standard", "regcls", "hybrid", "oracle", "siglip", "ensemble", "full"])
    ap.add_argument("--ens_alpha", type=float, default=0.5, help="geometric weight of siglip vs bridge")
    ap.add_argument("--sig_res", type=int, default=384, help="SigLIP vision input resolution")
    ap.add_argument("--adapter", default=None, help="path to trained SigLIP adapter")
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--limit", type=int, default=0, help="0 = full val set")
    ap.add_argument("--start", type=int, default=0, help="start index (disjoint tuning/report splits)")
    ap.add_argument("--soft", action="store_true", help="soft region composition instead of hard argmax")
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="extra multiplier on the logit_scale-weighted class term")
    args = ap.parse_args()
    print("CONFIG " + " ".join(f"{k}={v}" for k, v in sorted(vars(args).items())), flush=True)

    dev = "cuda"
    tdata = torch.load(args.text)
    text_all = F.normalize(tdata["embeddings"], dim=-1).to(dev)

    model = DinoRTPSeg(args.variant, args.weights, img_size=args.size, k=args.k, m=args.m, use_kernel=args.kernel).to(dev).eval()
    ck = torch.load(args.ckpt, map_location="cpu")
    missing = model.load_state_dict(ck["model"], strict=False)
    print(f"loaded {args.ckpt} (iter {ck.get('it','?')})")

    sig = None
    adapter = None
    if args.readout in ("siglip", "ensemble"):
        from siglip_region import SigLIPRegionClassifier
        sig = SigLIPRegionClassifier("./siglip_vision", res=args.sig_res).to(dev)
        if args.adapter:
            from train_adapter import Adapter
            adapter = Adapter().to(dev).eval()
            adapter.load_state_dict(torch.load(args.adapter, map_location=dev))

    ds = ADESemSeg(args.ade, "validation", args.size, native_gt=True)
    end = min(len(ds), args.start + args.limit) if args.limit else len(ds)
    n = end - args.start
    inter = torch.zeros(150, dtype=torch.float64)
    union = torch.zeros(150, dtype=torch.float64)
    correct = total = 0
    t0 = time.time()

    with torch.no_grad(), torch.amp.autocast("cuda", torch.bfloat16):
        for i in range(args.start, end):
            x, y = ds[i]
            x = x[None].to(dev)
            out = model(x, text_emb=text_all, refine=True)
            if args.readout in ("regcls", "hybrid"):
                rt = F.normalize(model.router.to_text(out["regions"][0]).float(), dim=-1)
                pcls = torch.softmax(rt @ text_all.t().float()
                                     * model.logit_scale.exp().float(), dim=-1)  # (K,150)
            if args.readout in ("siglip", "ensemble"):
                # zero-shot: SigLIP mask attention pooling per region ->
                # patch-level hard region assignment carries the class
                if adapter is not None:
                    emb = sig.region_embeddings(x, out["region_masks"][0].float(),
                                                out["grid"])
                    pcls = torch.softmax(adapter(emb) @ text_all.t().float()
                                         * adapter.logit_scale.exp().clamp(max=100),
                                         dim=-1)                     # (K,150)
                else:
                    pcls = sig.classify(x, out["region_masks"][0].float(),
                                        out["grid"], text_all)      # (K,150)
                if args.readout in ("ensemble", "full"):
                    # FC-CLIP-style geometric ensemble with the trained bridge
                    rt = F.normalize(model.router.to_text(out["regions"][0]).float(), dim=-1)
                    pbridge = torch.softmax(rt @ text_all.t().float()
                                            * model.logit_scale.exp().float(), dim=-1)
                    a = args.ens_alpha
                    pcls = (pcls.clamp(min=1e-8) ** a) * (pbridge.clamp(min=1e-8) ** (1 - a))
                M = out["region_masks"][0].float()
                h, w_ = out["grid"]
                if args.soft:
                    patch_score = M.t() @ pcls                      # (N,150) soft composition
                else:
                    patch_score = pcls[M.argmax(0)]                 # (N,150) hard assignment
                score = torch.log(patch_score.clamp(min=1e-8)).t().reshape(150, h, w_)
                if args.readout == "full":
                    # + kernel-composed per-class mask logits (best mask quality)
                    mask_hi = out["mask_hi"].float()
                    score = F.interpolate(score[None], size=mask_hi.shape[-2:],
                                          mode="bilinear")[0] + mask_hi
            elif args.readout == "oracle":
                # region-purity upper bound: hard-assign each patch to its
                # argmax region, give each region its GT majority class.
                # Bypasses reg_hi entirely -> measures proposal quality only.
                yp = F.interpolate(y[None, None].float().to(dev), size=out["grid"],
                                   mode="nearest")[0, 0].long().flatten()
                onehot = F.one_hot(yp, 151).float()
                M = out["region_masks"][0].float()                            # (K,N)
                soft = M @ onehot[:, 1:]                                      # (K,150)
                reg_cls = soft.argmax(-1)                                     # (K,)
                patch_cls = reg_cls[M.argmax(0)]                              # (N,)
                h, w_ = out["grid"]
                score = F.one_hot(patch_cls, 150).float().t().reshape(150, h, w_)
            elif args.readout == "regcls":
                # v2 readout: per-region 150-way softmax composed by region masks
                reg_prob = torch.sigmoid(out["region_masks_hi"].float())         # (K,160,160)
                score = torch.log(torch.einsum("kc,khw->chw", pcls, reg_prob)
                                  .clamp(min=1e-6))
            elif args.readout == "hybrid":
                # kernel-composed masks (mask quality) + region-CE class evidence
                mask_hi = out["mask_hi"].float()                                 # (150,160,160)
                masks = out["region_masks"][0].float()                           # (K,N)
                area = masks.sum(-1) / masks.sum()                               # (K,)
                cls_ev = torch.log((pcls * area[:, None]).sum(0).clamp(min=1e-6))  # (150,)
                score = mask_hi + args.alpha * cls_ev[:, None, None]
            else:
                mask_hi = out["mask_hi"].float()                   # (150,160,160)
                zt = F.normalize(out["text_embed"].float(), dim=-1)
                cls_logit = (zt @ text_all.t().float()).diag()     # (150,)
                pres = torch.sigmoid(out["presence"].float())      # C4 gating
                score = (mask_hi
                         + args.alpha * model.logit_scale.exp().float() * cls_logit[:, None, None]
                         + torch.log(pres.clamp(min=1e-6))[:, None, None])
            Hn, Wn = y.shape[-2:]                              # native GT resolution
            score = F.interpolate(score[None], size=(Hn, Wn), mode="bilinear")[0]
            pred = score.argmax(0).cpu()                           # 0..149
            yv = y - 1                                             # -1=ignore, 0..149
            valid = yv >= 0
            correct += int((pred[valid] == yv[valid]).sum())
            total += int(valid.sum())
            for c in torch.unique(torch.cat([pred[valid], yv[valid]])):
                pi, gi = pred[valid] == c, yv[valid] == c
                inter[c] += int((pi & gi).sum())
                union[c] += int((pi | gi).sum())
            if (i + 1 - args.start) % 200 == 0:
                iou = (inter / union.clamp(min=1)).numpy()
                seen = union.numpy() > 0
                print(f"[{i+1-args.start}/{n}] running mIoU {100*iou[seen].mean():.2f} "
                      f"acc {100*correct/max(total,1):.2f}", flush=True)

    iou = (inter / union.clamp(min=1)).numpy()
    seen = union.numpy() > 0
    dt = time.time() - t0
    print(f"FINAL mIoU {100*iou[seen].mean():.2f} | pixel acc "
          f"{100*correct/max(total,1):.2f} | {n/dt:.1f} img/s over {n} images")
    names = tdata["names"]
    order = np.argsort(iou)
    print("worst:", [(names[j], round(100*iou[j],1)) for j in order[:8] if seen[j]])
    print("best :", [(names[j], round(100*iou[j],1)) for j in order[::-1][:8]])


if __name__ == "__main__":
    main()
