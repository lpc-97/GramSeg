"""Weight surgery: official SED ckpt with its CLIP visual swapped back to LAION.

Base = official sed_convnextB.pth (everything: decoder/aggregator, CLIP text).
Swap = every 'sem_seg_head.predictor.clip_model.visual.*' key replaced by the
value from the frozen-training ckpt (model_final.pth), whose CLIP visual was
never trained and therefore still holds the original LAION weights.

Key selection follows sed_model.py's finetune gate: parameters of
sem_seg_head.predictor.clip_model with "visual" in the name are the (only)
CLIP weights that clip_finetune touches; the text tower / token_embedding /
logit_scale are always frozen. Verified on sed_convnextB.pth:
608 model keys total = 343 clip_model.visual.* + 150 clip_model.(text) + 115 rest.

Both ckpts are detectron2 format {model, trainer, iteration}; only 'model' is
read/modified, and the output contains only {'model': ...} (eval-only load
via DetectionCheckpointer needs nothing else).

Prints a per-group diff between the two ckpts first (sanity: CLIP text keys
should be identical in both — never trained anywhere; CLIP visual should
differ — official fine-tuned it; the rest will differ too because the frozen
run trained its own decoder — irrelevant, we keep the official ones).
"""
import argparse
import os

import torch

VISUAL_PREFIX = "sem_seg_head.predictor.clip_model.visual."
CLIP_PREFIX = "sem_seg_head.predictor.clip_model."


def load_model_sd(path):
    ck = torch.load(os.path.expanduser(path), map_location="cpu")
    if isinstance(ck, dict) and "model" in ck:
        print(f"[surgery] {path}: d2 ckpt, extra keys "
              f"{sorted(k for k in ck if k != 'model')}, "
              f"iteration={ck.get('iteration', '?')}")
        return ck["model"]
    print(f"[surgery] {path}: raw state_dict")
    return ck


def group_of(key):
    if key.startswith(VISUAL_PREFIX):
        return "clip_visual"
    if key.startswith(CLIP_PREFIX):
        return "clip_text"
    return "rest"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--official",
                    default="~/offline_staging/sed/sed_convnextB.pth",
                    help="base ckpt (fine-tuned CLIP visual + official rest)")
    ap.add_argument("--frozen",
                    default="~/SED/output/frozenB_20k/model_final.pth",
                    help="donor ckpt whose CLIP visual = untouched LAION")
    ap.add_argument("--out",
                    default="~/SED/output/step2/surgery_laionclip_officialrest.pth")
    args = ap.parse_args()
    out_path = os.path.expanduser(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    off = load_model_sd(args.official)
    frz = load_model_sd(args.frozen)

    # ---------------------------------------------------------- diff stats
    off_keys, frz_keys = set(off), set(frz)
    only_off, only_frz = sorted(off_keys - frz_keys), sorted(frz_keys - off_keys)
    common = sorted(off_keys & frz_keys)
    print(f"[diff] official {len(off_keys)} keys | frozen {len(frz_keys)} keys "
          f"| common {len(common)} | only-official {len(only_off)} "
          f"| only-frozen {len(only_frz)}")
    for name, keys in (("only-official", only_off), ("only-frozen", only_frz)):
        for k in keys[:10]:
            print(f"[diff]   {name}: {k}")
        if len(keys) > 10:
            print(f"[diff]   {name}: ... {len(keys) - 10} more")

    stats = {}  # group -> [n_common, n_identical, n_shape_mismatch]
    for k in common:
        g = stats.setdefault(group_of(k), [0, 0, 0])
        g[0] += 1
        if off[k].shape != frz[k].shape:
            g[2] += 1
        elif torch.equal(off[k], frz[k]):
            g[1] += 1
    for name in ("clip_visual", "clip_text", "rest"):
        n, same, bad = stats.get(name, (0, 0, 0))
        print(f"[diff] {name:12s}: {n:4d} common, {same:4d} identical, "
              f"{n - same - bad:4d} differ, {bad} shape-mismatch")
    n, same, _ = stats.get("clip_text", (0, 0, 0))
    if n and same != n:
        print("[diff] WARNING: CLIP text keys differ between ckpts — "
              "unexpected (text tower is never trained). Check the ckpts.")
    n, same, _ = stats.get("clip_visual", (0, 0, 0))
    if n and same == n:
        print("[diff] WARNING: all CLIP visual keys already identical — "
              "surgery would be a no-op. Check the ckpts.")

    # ------------------------------------------------------------- surgery
    new_sd = dict(off)  # official is the base
    replaced = changed = 0
    missing = []
    for k in off_keys:
        if not k.startswith(VISUAL_PREFIX):
            continue
        if k not in frz:
            missing.append(k)
            continue
        assert off[k].shape == frz[k].shape, \
            f"shape mismatch on {k}: {tuple(off[k].shape)} vs {tuple(frz[k].shape)}"
        if not torch.equal(off[k], frz[k]):
            changed += 1
        new_sd[k] = frz[k].clone()
        replaced += 1
    assert not missing, f"visual keys absent from frozen ckpt: {missing[:5]}"
    assert replaced > 0, "no CLIP visual keys found — wrong ckpt structure?"
    print(f"[surgery] replaced {replaced} clip_model.visual.* keys "
          f"({changed} actually differed) out of {len(off_keys)} total")

    torch.save({"model": new_sd}, out_path)
    print(f"SURGERY_DONE -> {out_path}")


if __name__ == "__main__":
    main()
