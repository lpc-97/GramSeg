"""Precompute SigLIP text embeddings for ADE20K-150 class names (offline).

Reads objectInfo150.txt from the ADE20K root, averages embeddings over
synonyms and prompt templates, saves (150, 1152) tensor + names.

Usage: python precompute_text.py --ade data/ADEChallengeData2016 --out ade150_text.pt
"""
import argparse, torch

TEMPLATES = [
    "a photo of a {}.",
    "a photo of the {}.",
    "there is a {} in the scene.",
    "a bad photo of a {}.",
    "{}.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ade", required=True)
    ap.add_argument("--out", default="ade150_text.pt")
    ap.add_argument("--model", default="google/siglip-so400m-patch14-384")
    args = ap.parse_args()

    from transformers import AutoTokenizer, SiglipTextModel
    tok = AutoTokenizer.from_pretrained(args.model)
    txt = SiglipTextModel.from_pretrained(args.model).cuda().eval()

    if args.ade == "coco171":
        from coco_names import NAMES_171
        names = list(NAMES_171)
    else:
        names = []
        with open(f"{args.ade}/objectInfo150.txt") as f:
            next(f)                                    # header: Idx Ratio Train Val Name
            for line in f:
                parts = line.strip().split("\t")
                names.append(parts[-1])                # e.g. "wall" or "bed;beds"

    embs = []
    with torch.no_grad():
        for name in names:
            syns = [s.strip() for s in name.replace(";", ",").split(",") if s.strip()]
            prompts = [t.format(s) for s in syns for t in TEMPLATES]
            batch = tok(prompts, padding="max_length", max_length=64,
                        truncation=True, return_tensors="pt").to("cuda")
            e = txt(**batch).pooler_output                 # (n, 1152)
            e = torch.nn.functional.normalize(e, dim=-1).mean(0)
            embs.append(torch.nn.functional.normalize(e, dim=-1))
    embs = torch.stack(embs).cpu()                     # (150, 1152)
    torch.save({"embeddings": embs, "names": names}, args.out)
    print(f"saved {args.out}: {tuple(embs.shape)}, first classes: {names[:5]}")


if __name__ == "__main__":
    main()
