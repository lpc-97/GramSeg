# Patches against upstream CorrCLIP

Upstream ships no LICENSE file, so its code is "all rights reserved" and cannot be
redistributed — not even as a patched copy. These diffs are our own contribution;
apply them to a checkout you obtain yourself.

## Pinned upstream

```bash
git clone https://github.com/zdk258/CorrCLIP
cd CorrCLIP
git checkout 6c1fd2c02ff90f071ed744a5998be0cda7ad75b4
```

That is the commit our diffs were generated against (recorded from the `origin/master`
ref of the snapshot used for every reported run). Later upstream commits may not
apply cleanly.

## Apply

```bash
# from the CorrCLIP checkout root
git apply --check /path/to/GramSeg/patches/corrclip_segmentor_gram.patch      # Gram mask source
git apply         /path/to/GramSeg/patches/corrclip_segmentor_gram.patch

git apply /path/to/GramSeg/patches/corrclip_segmentor_dinov3.patch            # DINOv3 backbone wiring
git apply /path/to/GramSeg/patches/corrclip_dinov3_upscale.patch              # patch-16 token upscaling
```

Then make the operator importable, e.g.:

```bash
export PYTHONPATH=/path/to/GramSeg/gramseg:$PYTHONPATH
```

## What each patch does

| Patch | +/- lines | Purpose |
|---|---|---|
| `corrclip_segmentor_gram.patch` | +28 / -1 | Registers `mask_generator='gram'` so the segmentor calls our operator instead of loading SAM2. |
| `corrclip_segmentor_dinov3.patch` | +58 / -6 | Wires a frozen DINOv3 backbone (patch 16) as the token source. |
| `corrclip_dinov3_upscale.patch` | +21 / -3 | Token-grid upscaling so patch-16 geometry lines up with the evaluation protocol. |

## Large-vocabulary path

`apply_memfix_patch.py` edits the segmentor in place to add the direct-label
interface used by the vocabulary-scaling study (`GRAM_VOCAB_MEMFIX=1`). It is a
script rather than a diff because it locates its two insertion points by pattern.

```bash
python apply_memfix_patch.py /path/to/CorrCLIP/corrclip_segmentor.py
```

The path argument is optional and defaults to `~/CorrCLIP/corrclip_segmentor.py`,
the location used in our runs. Run it after the three diffs above. It prints
`PATCH OK` on success and raises `AssertionError` if the file is already patched or
if either anchor is missing (which means the checkout is not the pinned commit) —
so it is safe to re-run, but it is not silent.

When `GRAM_VOCAB_MEMFIX` is unset, the patched code path is byte-for-byte the
original one.
