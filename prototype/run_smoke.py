"""Smoke test + latency benchmark for DINO-RTP-Seg on GPU.

Usage: python run_smoke.py --weights dinov3_vits16.safetensors [--variant vits16]
"""
import argparse, time, torch
from dino_rtp_seg import DinoRTPSeg


def bench(fn, warmup=5, iters=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000  # ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--variant", default="vits16")
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--fp16", action="store_true")
    args = ap.parse_args()

    dev = "cuda"
    model = DinoRTPSeg(args.variant, args.weights, img_size=args.size,
                       k=args.k).to(dev).eval()
    n_train = sum(p.numel() for p in model.trainable_parameters())
    n_total = sum(p.numel() for p in model.parameters())
    print(f"variant={args.variant} size={args.size} K={args.k}")
    print(f"params: total={n_total/1e6:.1f}M trainable={n_train/1e6:.2f}M "
          f"frozen-backbone={ (n_total-n_train)/1e6:.1f}M")

    x = torch.randn(1, 3, args.size, args.size, device=dev)
    text = torch.randn(150, 1152, device=dev)      # fake ADE20K-150 vocab
    pts = torch.rand(3, 2, device=dev)
    ex = torch.tensor([[0.2, 0.2, 0.6, 0.7]], device=dev)

    ctx = torch.autocast("cuda", torch.float16) if args.fp16 else torch.no_grad()
    with torch.no_grad(), ctx if args.fp16 else torch.no_grad():
        out = model(x, text_emb=text, points=pts, exemplar_boxes=ex)
        print("smoke OK:",
              "mask_patch", tuple(out["mask_patch"].shape),
              "mask_hi", tuple(out["mask_hi"].shape),
              "presence", tuple(out["presence"].shape),
              "kinds", len(out["kinds"]))

        # component latency
        feat, taps, grid = model.backbone(x)
        t_bb = bench(lambda: model.backbone(x))
        t_prop = bench(lambda: model.proposal(feat))
        masks, regions = model.proposal(feat)
        temb = model.prompt_enc.encode_text(text)
        t_route = bench(lambda: model.router(temb, regions, masks))
        t_ref = bench(lambda: model.refine(masks[0], taps, grid)[0])
        t_e2e = bench(lambda: model(x, text_emb=text))
        t_int = bench(lambda: model(x, points=pts))

    print(f"[latency ms @1x3090] backbone={t_bb:.1f} gram-seed={t_prop:.1f} "
          f"route(150cls)={t_route:.2f} refine={t_ref:.1f}")
    print(f"end2end semantic(150cls)={t_e2e:.1f}ms ({1000/t_e2e:.1f} FPS) | "
          f"interactive(3pts)={t_int:.1f}ms ({1000/t_int:.1f} FPS)")

    # vocabulary scaling (C3 claim: sub-linear in vocab size)
    for v in (20, 150, 847):
        tv = torch.randn(v, 1152, device=dev)
        with torch.no_grad():
            t = bench(lambda: model(x, text_emb=tv), warmup=3, iters=15)
        print(f"vocab={v:4d}: {t:.1f}ms ({1000/t:.1f} FPS)")


if __name__ == "__main__":
    main()
