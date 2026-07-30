# GramSeg

Training-free open-vocabulary semantic segmentation whose region proposals come from
**Gram-affinity propagation over frozen DINOv3 patch tokens** — no segmentation
foundation model (SAM/SAM2), no diffusion prior, no external data, no training, and
no new learnable parameters.

The whole method is one module: frozen ViT patch tokens → farthest-point-sampling
seeds → a *one-step* Gram affinity softmax → a binarized region map that a frozen
CLIP then classifies at the region level.

```
gramseg/gram_mask_generator.py     <- the entire operator (~29 KB, pure torch)
```

It depends only on `torch`, `numpy`, `PIL` and uses only `matmul / softmax /
interpolate / sort`, so it drops into an existing pipeline without pulling anything
else along.

## Status

> **The paper is under review. This repository currently contains code only.**
>
> Benchmark tables, per-run evaluation logs, the number-provenance ledger, and the
> figure-generating scripts are deliberately **not** included yet — they will be
> added here once the paper is public. Nothing in this repo should be read as a
> reported result.

## Why patches instead of a fork

Our evaluation harness is [CorrCLIP](https://github.com/zdk258/CorrCLIP)'s. The
upstream repository ships **no top-level LICENSE file**, so its source is "all
rights reserved" and we may not redistribute it — not even a patched copy.

So this repository contains **zero CorrCLIP source**. Instead it ships our own
modules plus minimal unified diffs against a pinned upstream commit
(`6c1fd2c02ff90f071ed744a5998be0cda7ad75b4`). You clone upstream yourself and apply
them. See [`patches/README.md`](patches/README.md).

## Repository layout

| Path | What it is |
|---|---|
| `gramseg/gram_mask_generator.py` | The core operator. Original work, Apache-2.0. |
| `gramseg/gen_gram_masks.py` | Offline route: pre-generate `<stem>.npz` region maps. |
| `gramseg/run_gram_eval.sh` | Single-config evaluation driver. |
| `patches/*.patch` | Our diffs against upstream CorrCLIP (21/58/28 added lines). |
| `patches/apply_memfix_patch.py` | Programmatic patch enabling the large-vocabulary label path. |
| `bench/` | Benchmark preparation (dataset conversion, sweep drivers). |
| `experiments/` | The orchestration scripts as actually run, verbatim. |
| `prototype/` | Earlier standalone prototype (before adopting the CorrCLIP harness). |
| `legacy/sed_frozen/` | An abandoned exploration (frozen-backbone SED adapters). Kept for the record; not part of the method. |

## Two ways to use it

**Route A — offline.** Pre-generate region maps, then point the harness at them.
The output contract matches CorrCLIP's pre-generated `region_masks` npz exactly
(one `int16` map per image, `0` = unsegmented union, ids `1..M` under 10000).

```bash
python gramseg/gen_gram_masks.py \
  --images /path/to/val_images --out /path/to/region_masks_gram \
  --weights /path/to/dinov3_vitb16.safetensors --variant vitb16 \
  --k 64 --img-size 896 --upsample nearest \
  --multiscale --coarse-size 448 --merge-thresh 0.6 \
  --pamr-iters 10 --pamr-res 1024
```

`--variant` accepts `vits16` / `vitb16` (DINOv3, patch 16) and
`dinov2_vits14` / `dinov2_vitb14` (DINOv2, kept as a backbone-attribution control).

**Route B — online.** Apply the patches, then select the mask source at eval time:

```bash
GRAM_MCT_DILATE=3 python eval.py --config ./configs/cfg_ade20k.py \
  --cfg-options model.mask_generator=gram model.model_type=ViT-L-14-quickgelu
```

Add `GRAM_MS=0 GRAM_PAMR=0` for the single-forward fast mode (no multi-scale, no
PAMR refinement).

## Environment knobs

Read by the operator / the patched segmentor:

| Variable | Meaning |
|---|---|
| `GRAM_MCT_DILATE` | Region-CLS context dilation radius. `3` in every reported configuration. |
| `GRAM_MS` | `0` disables multi-scale mask generation. |
| `GRAM_PAMR` | `0` disables PAMR refinement. `GRAM_PAMR_RES` dominates peak memory. |
| `GRAM_GEN` | Selects the online generator path. |
| `GRAM_VOCAB_MEMFIX` | Direct-label interface for large vocabularies (see `patches/apply_memfix_patch.py`). |
| `GRAM_K`, `GRAM_TAU`, `GRAM_REFINE`, `GRAM_TOP_P`, `GRAM_BINARIZE`, `GRAM_CONF`, `GRAM_IMG_SIZE`, `GRAM_UPSAMPLE`, `GRAM_VARIANT`, `GRAM_WEIGHTS` | Per-knob overrides mirroring the CLI flags above. |

Every reported configuration used `GRAM_MCT_DILATE=3`; the dilation radius was
frozen globally once chosen, and no per-dataset tuning was applied.

## Reproducing the runs

`experiments/` holds the batch scripts exactly as they ran on our cluster, so the
commands, seeds, and flags are auditable rather than paraphrased. They `source
~/night/common.sh`, a machine-local file that is **not** part of this repo because
it only contained paths; `experiments/common.sh.example` is a reconstruction of the
variables it must define.

Because they are verbatim, the scripts still contain cluster-specific assumptions
(GPU indices, `~/batchX` output dirs, an offline HF cache). Read before running.

## Environment

Follow upstream CorrCLIP's `requirements.txt` for the harness (mmsegmentation /
mmengine / open_clip). GramSeg itself adds no dependency beyond `torch >= 2.0`,
`numpy`, and `pillow`. Model weights (DINOv3, CLIP/MetaCLIP, and the baseline
SAM 2 / SD 2.1 / dino.txt checkpoints) must be obtained from their official
sources; see `NOTICE`.

## License

Apache License 2.0 — see [`LICENSE`](LICENSE). Third-party attribution and the
license status of every upstream component are documented in [`NOTICE`](NOTICE).

## Citation

Paper under review. A citation entry will be added here when it is public.
