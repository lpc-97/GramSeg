"""Gram-Seed mask generator for CorrCLIP (go/no-go experiment #2).

Replaces SAM2 as CorrCLIP's mask source: frozen ViT patch tokens (DINOv3 by
default; DINOv2 is supported as a backbone-attribution control, see
--variant below) -> FPS seeds -> one-step Gram affinity softmax ->
binarized region map.

Output contract (matches CorrCLIP's pre-generated region_masks npz exactly,
see corrclip_segmentor.py generate_mask() and eomt/generate_all_makss.py):
  * one int map of shape [H_orig, W_orig] per image
  * value 0 (the minimum) = union of unsegmented/low-confidence regions
  * values 1..M = region ids (M <= K, empty ids compacted away)
  * ids stay < 10000 (CorrCLIP uses ids >= 10000 internally for merged
    clusters and for the padding value in sliding-window inference)
  * saved as np.savez_compressed(<stem>.npz, instance_mask=<int16 array>)

This module is shared by:
  * gen_gram_masks.py           (offline route a: pre-generate npz files)
  * corrclip_segmentor.py       (online route b: mask_generator='gram')

torch >= 2.0 compatible: only uses matmul / softmax / interpolate / sort.
No dependency on the rest of the proto codebase.
"""

import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# timm model tag per --variant. DINOv3 variants ship as local safetensors
# (offline); DINOv2 variants have no local weights and fall back to timm's
# own pretrained=True hub download (see load_backbone below) -- this is the
# red-team "backbone attribution" control: same pipeline, older/no-register
# backbone, to show whether our gain is the method or just DINOv3.
TIMM_NAMES = {
    "vits16": "vit_small_patch16_dinov3.lvd1689m",
    "vitb16": "vit_base_patch16_dinov3.lvd1689m",
    "vitl16": "vit_large_patch16_dinov3.lvd1689m",
    "dinov2_vits14": "vit_small_patch14_dinov2.lvd142m",
    "dinov2_vitb14": "vit_base_patch14_dinov2.lvd142m",
}
# ViT patch size per variant -- all grid math below reads this per-instance
# value (self.patch), never a literal, so patch14 (DINOv2) and patch16
# (DINOv3) are both handled correctly.
PATCH_SIZES = {
    "vits16": 16,
    "vitb16": 16,
    "vitl16": 16,
    "dinov2_vits14": 14,
    "dinov2_vitb14": 14,
}


def load_backbone(variant="vits16", weights_file=None, device="cuda",
                   download=True):
    """timm ViT backbone: DINOv3 (local safetensors) or DINOv2 (timm hub).

    weights_file: path to a local timm-converted safetensors file. If it is
    given and exists on disk, it is always used (pretrained=True + a
    pretrained_cfg_overlay pointing at the file) regardless of variant.

    Otherwise:
      * DINOv3 variants (vits16/vitb16) have no online weights in this repo
        -> randomly initialized (smoke tests only), same as before.
      * DINOv2 variants (dinov2_vits14/dinov2_vitb14) ship no local weights
        -> timm pretrained=True, which downloads from the timm/HF hub (the
        caller is responsible for setting HF_ENDPOINT etc. for mirrors;
        this function does not touch networking config). Set download=False
        to force a random init instead (used by offline smoke tests so they
        never hit the network).

    Returns (model, mean, std).
    """
    import timm
    has_local = weights_file is not None and os.path.isfile(weights_file)
    is_dinov2 = variant.startswith("dinov2_")
    kwargs = dict(num_classes=0, dynamic_img_size=True)
    if has_local:
        kwargs["pretrained"] = True
        kwargs["pretrained_cfg_overlay"] = dict(file=weights_file)
    elif is_dinov2 and download:
        kwargs["pretrained"] = True          # timm hub auto-download
    else:
        kwargs["pretrained"] = False         # random init (smoke tests)
    model = timm.create_model(TIMM_NAMES[variant], **kwargs)
    model.eval().requires_grad_(False)
    model.to(device)
    try:
        cfg = timm.data.resolve_model_data_config(model)
        mean, std = tuple(cfg["mean"]), tuple(cfg["std"])
    except Exception:
        mean, std = IMAGENET_MEAN, IMAGENET_STD
    return model, mean, std


@torch.no_grad()
def farthest_point_sample(fhat, k):
    """fhat: (N, d) L2-normalized. Returns (k_eff,) seed indices, k_eff=min(k,N)."""
    n = fhat.shape[0]
    k = min(k, n)
    idx = torch.zeros(k, dtype=torch.long, device=fhat.device)
    center = F.normalize(fhat.mean(0, keepdim=True), dim=-1)
    dist = 1.0 - (fhat @ center.t()).squeeze(-1)          # (N,)
    idx[0] = dist.argmax()
    min_d = torch.full((n,), 1e9, device=fhat.device, dtype=fhat.dtype)
    for i in range(1, k):
        d = 1.0 - fhat @ fhat[idx[i - 1]]
        min_d = torch.minimum(min_d, d)
        idx[i] = min_d.argmax()
    return idx


@torch.no_grad()
def spherical_kmeans_probs(feat, k=64, iters=15, seed=0):
    """Trivial-clustering control: spherical k-means on the SAME tokens.

    Random init (fixed seed, CPU RNG so it is device-independent), Lloyd
    updates under cosine distance (assign = argmax cosine to centroid,
    update = renormalized mean), early stop on an assignment fixpoint.
    Empty clusters keep their previous centroid and are compacted away
    downstream. Returns a one-hot (K_eff, N) hard assignment matrix in the
    same (columns sum to 1) format as gram_seed_probs, so every decision
    path (hard/topp, nearest/soft, PAMR) works unchanged.

    Contrast with gram_seed_probs (FPS init + single-step propagation):
    the diff between the two isolates whether our seed selection + one-step
    Gram propagation adds anything over vanilla clustering."""
    fhat = F.normalize(feat, dim=-1)
    n = fhat.shape[0]
    k = min(k, n)
    g = torch.Generator().manual_seed(seed)
    init = torch.randperm(n, generator=g)[:k].to(fhat.device)
    cent = fhat[init]                                      # (K, d)
    assign = None
    for _ in range(iters):
        new_assign = (cent @ fhat.t()).argmax(0)           # (N,) cosine
        if assign is not None and torch.equal(new_assign, assign):
            break
        assign = new_assign
        onehot = F.one_hot(assign, k).t().to(fhat.dtype)   # (K, N)
        counts = onehot.sum(1, keepdim=True)
        cent = torch.where(counts > 0,
                           F.normalize(onehot @ fhat, dim=-1), cent)
    return F.one_hot(assign, k).t().to(feat.dtype)


@torch.no_grad()
def gram_seed_probs(feat, k=64, tau=0.07, refine_rounds=1):
    """feat: (N, d) fp32 patch tokens.

    Returns probs (K_eff, N): per-token softmax over regions (columns sum to 1).
    Same math as proto GramSeedProposal minus the (trainable) TinyFormer,
    which is untrained here and must be skipped for zero-shot masks.
    """
    fhat = F.normalize(feat, dim=-1)
    seeds = farthest_point_sample(fhat, k)
    q = fhat[seeds]                                        # (K, d)
    probs = torch.softmax(q @ fhat.t() / tau, dim=0)       # (K, N)
    for _ in range(refine_rounds):
        regions = (probs @ feat) / probs.sum(-1, keepdim=True).clamp(min=1e-6)
        rhat = F.normalize(regions, dim=-1)
        probs = torch.softmax(rhat @ fhat.t() / tau, dim=0)
    return probs


def binarize_hard(probs, conf_thresh=0.0):
    """Mutually exclusive partition: per-token argmax.

    Returns labels (N,) int64; 0 = unassigned (only when conf_thresh > 0),
    1..K = region id (not yet compacted).
    """
    conf, labels = probs.max(0)
    labels = labels + 1
    if conf_thresh > 0:
        labels = labels.masked_fill(conf < conf_thresh, 0)
    return labels


def topp_membership(probs, top_p=0.9):
    """Per-token nucleus: smallest set of regions covering cumulative prob top_p.

    Returns member (K, N) bool, possibly overlapping (SAM2-style region sets).
    The argmax region is always a member, so every token is covered.
    """
    sp, si = probs.sort(0, descending=True)
    cum = sp.cumsum(0)
    keep_sorted = (cum - sp) < top_p                       # first row always True
    member = torch.zeros_like(keep_sorted)
    member.scatter_(0, si, keep_sorted)
    return member


def paint_by_area(member):
    """Paint possibly-overlapping binary masks into one int map.

    Ascending area order, exactly like CorrCLIP's sam2 branch: larger
    regions painted later, so they win overlaps. member: (K, N) bool.
    Returns labels (N,) int64 with ids 1..M (empty masks skipped).
    """
    areas = member.sum(1)
    ids = torch.nonzero(areas > 0).squeeze(1)
    order = torch.argsort(areas[ids])                      # ascending
    ids = ids[order]
    labels = torch.zeros(member.shape[1], dtype=torch.long, device=member.device)
    for new_id, k_idx in enumerate(ids.tolist(), start=1):
        labels[member[k_idx]] = new_id
    return labels


def compact_labels(labels):
    """Remove empty ids, remap present ids to 1..M keeping 0 as background."""
    vals = torch.unique(labels)
    vals = vals[vals > 0]
    out = torch.zeros_like(labels)
    for new_id, v in enumerate(vals.tolist(), start=1):
        out[labels == v] = new_id
    return out


# --------------------------------------------------------------------------- #
# PAMR: pixel-adaptive mask refinement (Araslanov & Roth, CVPR'20; used
# training-free by NACLIP/ClearCLIP). Pure conv, no parameters: iteratively
# re-averages each region prob map over a multi-dilation 3x3 neighborhood,
# weighted by RGB affinity, so patch-grid masks snap to image edges.
# --------------------------------------------------------------------------- #

_PAMR_POS = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)]


def _neighbor_kernel(device, dtype):
    """(8,1,3,3) conv kernels, each extracting one 3x3 neighbor (no center)."""
    k = torch.zeros(8, 1, 3, 3, device=device, dtype=dtype)
    for i, (y, x) in enumerate(_PAMR_POS):
        k[i, 0, y, x] = 1.0
    return k


def _unfold_neighbors(x, kern, dilation):
    """x (B,C,H,W) -> (B,C,8,H,W): the 8 neighbors at the given dilation
    (zero padded at borders)."""
    b, c, h, w = x.shape
    y = F.conv2d(x, kern.repeat(c, 1, 1, 1), padding=dilation,
                 dilation=dilation, groups=c)
    return y.view(b, c, 8, h, w)


@torch.no_grad()
def pamr(image, probs, iters=10, dilations=(1, 2, 4, 8, 12, 24), chunk=16):
    """image: (1,3,H,W) RGB in [0,1]. probs: (1,K,H,W) per-pixel region
    distribution (channels sum to 1). Returns refined (1,K,H,W).

    Affinity (computed once): softmax over the 8*len(dilations) neighbors of
    -|I_p - I_n| / (eps + 0.1 * local_std), averaged over RGB — exactly the
    original PAMR. Each channel is then smoothed independently with the same
    weights, so channel sums (=1) are preserved and the argmax label map
    aligns to image edges. Chunked over K to bound memory at full res."""
    kern = _neighbor_kernel(image.device, image.dtype)
    nbs = [_unfold_neighbors(image, kern, d) for d in dilations]
    nb = torch.cat(nbs, dim=2)                              # (1,3,8D,H,W)
    std = torch.cat([nb, image.unsqueeze(2)], dim=2).std(2, keepdim=True)
    aff = -(nb - image.unsqueeze(2)).abs() / (1e-8 + 0.1 * std)
    aff = aff.mean(1, keepdim=True)                         # (1,1,8D,H,W)
    aff = torch.softmax(aff, dim=2)
    del nb, nbs, std
    k = probs.shape[1]
    for _ in range(iters):
        out = torch.empty_like(probs)
        for s in range(0, k, chunk):
            p = probs[:, s:s + chunk]
            n = torch.cat([_unfold_neighbors(p, kern, d) for d in dilations],
                          dim=2)                            # (1,c,8D,H,W)
            out[:, s:s + chunk] = (n * aff).sum(2)
        probs = out
    # zero padding attenuates all channels near borders by the same factor;
    # renormalizing the channel sum restores an exact per-pixel distribution
    return probs / probs.sum(1, keepdim=True).clamp(min=1e-6)


@torch.no_grad()
def merge_probs_multiscale(probs_f, feat_f, grid_f, probs_c, grid_c,
                           merge_thresh=0.6):
    """Hierarchically merge fine-scale regions under coarse-scale regions.

    probs_f: (Kf, Nf) fine-pass region distribution, feat_f: (Nf, d) fine
    tokens, probs_c: (Kc, Nc) coarse-pass distribution. The coarse label map
    is upsampled to the fine grid; each fine seed is assigned to the coarse
    region it (softly) overlaps most. A fine seed is absorbed iff the cosine
    between its region mean and its coarse group's mean (both in fine feature
    space) >= merge_thresh — coarse containment supplies the whole-object
    grouping, the threshold guards against coarse under-segmentation leaks.

    Merging seeds = summing their softmax rows, so the result (G, Nf) is
    still a per-token distribution and feeds every downstream path
    (hard/topp, nearest/soft, PAMR) unchanged."""
    ghc, gwc = grid_c
    labels_c = binarize_hard(probs_c)                       # (Nc,) in 1..Kc
    lc = labels_c.view(1, 1, ghc, gwc).float()
    lc = F.interpolate(lc, size=grid_f, mode="nearest")[0, 0].long().view(-1)
    kc = int(lc.max())
    onehot = F.one_hot(lc - 1, kc).to(probs_f.dtype)        # (Nf, Kc)

    overlap = probs_f @ onehot                              # (Kf, Kc)
    group = overlap.argmax(1)                               # (Kf,)

    denom = probs_f.sum(1, keepdim=True).clamp(min=1e-6)
    rf = F.normalize((probs_f @ feat_f) / denom, dim=-1)    # (Kf, d)
    w = onehot / onehot.sum(0).clamp(min=1)
    rc = F.normalize(w.t() @ feat_f, dim=-1)                # (Kc, d)
    sim = (rf * rc[group]).sum(-1)                          # (Kf,)

    kf = probs_f.shape[0]
    own = kc + torch.arange(kf, device=probs_f.device)
    new_id = torch.where(sim >= merge_thresh, group, own)
    uniq, inv = torch.unique(new_id, return_inverse=True)   # compact group ids
    merged = torch.zeros(uniq.numel(), probs_f.shape[1],
                         device=probs_f.device, dtype=probs_f.dtype)
    merged.index_add_(0, inv, probs_f)
    return merged


@torch.no_grad()
def upsample_probs_argmax(probs, grid, size, chunk=32):
    """Soft upsample: bilinear on the (K, gh, gw) prob maps, chunked over K to
    bound memory, then per-pixel argmax. Returns (conf, arg) each (H, W)."""
    gh, gw = grid
    best, arg = None, None
    for s in range(0, probs.shape[0], chunk):
        p = probs[s:s + chunk].view(1, -1, gh, gw)
        up = F.interpolate(p, size=size, mode="bilinear", align_corners=False)[0]
        m, a = up.max(0)
        if best is None:
            best, arg = m, a + s
        else:
            swap = m > best
            best = torch.where(swap, m, best)
            arg = torch.where(swap, a + s, arg)
    return best, arg


class GramMaskGenerator:
    """generate(PIL.Image) -> int32 tensor [H_orig, W_orig] on `device`.

    variant: 'vits16'/'vitb16' (DINOv3, patch16, 5 prefix tokens, local
             safetensors weights) or 'dinov2_vits14'/'dinov2_vitb14'
             (DINOv2, patch14, 1 prefix token/no registers, timm hub
             pretrained=True unless a local weights_file is given) -- the
             DINOv2 variants exist as a backbone-attribution control (same
             pipeline, older backbone). Patch size and prefix-token count
             are both read from the loaded model/variant, never hardcoded.
    download: for DINOv2 variants with no usable local weights_file,
              whether to let timm auto-download pretrained weights from the
              hub (default True). Set False to force a random-init backbone
              (used by offline smoke tests).
    binarize: 'hard' (mutually exclusive argmax partition) or
              'topp'  (per-token nucleus sets, painted by ascending area)
    upsample: 'nearest' (patch-grid labels, nearest to full res) or
              'soft'    (bilinear on probs/masks, decide at full res)
    conf_thresh: tokens/pixels whose max region prob is below this become 0
                 (unsegmented union — the semantics CorrCLIP gives to the
                 minimum mask value). 0.0 = full partition, no background.

    Training-free refinements (all default OFF = bit-identical to baseline):
    multiscale:  extra coarse pass at coarse_size; fine regions merged under
                 coarse regions by feature similarity >= merge_thresh
                 (attacks over-segmentation of whole objects).
    pamr_iters:  >0 enables PAMR boundary refinement — probs are bilinearly
                 upsampled to pixel level, refined with RGB affinity, and the
                 hard/topp decision is taken at pixel level (replaces the
                 `upsample` setting for the decision; attacks 56x56 blur).
    attn_mix:    >0 blends token cosine with last-layer key-key cosine:
                 sim = (1-a)*cos(feat) + a*cos(key), via feature concat.

    Trivial-clustering controls (red-team baselines, default 'gram' = ours):
    cluster='kmeans': spherical k-means (random init, fixed seed, Lloyd,
                 cosine) on the SAME DINOv3 tokens -> hard partition.
    cluster='slic':  skimage SLIC superpixels (n_segments=K, RGB space,
                 no features at all) -> pure-image baseline. DINOv3 is not
                 loaded in this mode (weights_file ignored).
    """

    def __init__(self, weights_file, variant="vits16", k=64, tau=0.07,
                 refine_rounds=1, binarize="hard", top_p=0.9, conf_thresh=0.0,
                 img_size=896, upsample="nearest", device="cuda", fp16=True,
                 multiscale=False, coarse_size=0, coarse_k=0, merge_thresh=0.6,
                 pamr_iters=0, pamr_dilations=(1, 2, 4, 8, 12, 24), pamr_res=0,
                 attn_mix=0.0, cluster="gram", kmeans_iters=15, seed=0,
                 download=True):
        assert binarize in ("hard", "topp"), binarize
        assert upsample in ("nearest", "soft"), upsample
        assert cluster in ("gram", "kmeans", "slic"), cluster
        assert variant in PATCH_SIZES, f"unknown variant {variant!r}, " \
            f"choose one of {sorted(PATCH_SIZES)}"
        self.cluster, self.kmeans_iters, self.seed = cluster, kmeans_iters, seed
        self.k, self.tau, self.refine_rounds = k, tau, refine_rounds
        self.binarize, self.top_p, self.conf_thresh = binarize, top_p, conf_thresh
        self.img_size, self.upsample = img_size, upsample
        self.multiscale = multiscale
        self.coarse_size = coarse_size if coarse_size > 0 else img_size // 2
        self.coarse_k = coarse_k if coarse_k > 0 else max(8, k // 2)
        self.merge_thresh = merge_thresh
        self.pamr_iters = pamr_iters
        self.pamr_dilations = tuple(pamr_dilations)
        self.pamr_res = pamr_res
        self.attn_mix = attn_mix
        self.device = torch.device(device)
        self.fp16 = fp16 and self.device.type == "cuda"
        self.patch = PATCH_SIZES[variant]    # 16 for DINOv3, 14 for DINOv2
        # ViT-L DINOv3 overflows fp16 autocast; bf16 has fp32 range
        self.amp_dtype = torch.bfloat16 if variant == "vitl16" else torch.float16
        if cluster == "slic":                # image-only baseline: no backbone
            self.model, mean, std = None, IMAGENET_MEAN, IMAGENET_STD
        else:
            self.model, mean, std = load_backbone(
                variant, weights_file, self.device, download=download)
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)
        # read from the model, never hardcoded: DINOv3 = 5 (cls+4 registers),
        # DINOv2 = 1 (cls only, no registers).
        self.num_prefix = getattr(self.model, "num_prefix_tokens", 1)
        self._qkv_out = None
        if self.attn_mix > 0 and self.model is not None:
            def _hook(_m, _inp, out):        # capture last-block qkv output
                self._qkv_out = out
            self.model.blocks[-1].attn.qkv.register_forward_hook(_hook)

    def _preprocess(self, pil_img, img_size=None):
        """Resize longest side to img_size, snap both dims to multiples of
        self.patch (>= patch, handles sizes not divisible by the patch
        size)."""
        p = self.patch
        w, h = pil_img.size
        scale = float(img_size or self.img_size) / max(h, w)
        nh = max(p, int(round(h * scale / p)) * p)
        nw = max(p, int(round(w * scale / p)) * p)
        img = pil_img.resize((nw, nh), Image.BILINEAR)
        x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)
        x = x.permute(2, 0, 1)
        x = (x - self.mean) / self.std
        return x.unsqueeze(0), (nh // p, nw // p)

    @torch.no_grad()
    def _tokens(self, x, grid):
        if self.fp16:
            with torch.autocast("cuda", dtype=self.amp_dtype):
                t = self.model.forward_features(x)
        else:
            t = self.model.forward_features(x)
        t = t.float()[0]                                   # (prefix+N, d)
        n = grid[0] * grid[1]
        if t.shape[0] != n:                                # strip cls+registers
            # prefix count comes from the model (self.num_prefix), not a
            # hardcoded constant: 5 for DINOv3 (cls+4 registers), 1 for
            # DINOv2 (cls only). Assert it matches what forward_features
            # actually returned, so a variant/grid mismatch fails loudly.
            assert t.shape[0] - n == self.num_prefix, (
                f"prefix-token mismatch: forward_features returned "
                f"{t.shape[0]} tokens, expected {n} patch + "
                f"{self.num_prefix} prefix (grid={grid})")
            t = t[-n:]
        if self.attn_mix > 0:
            t = self._mix_keys(t, n)
        return t

    def _mix_keys(self, feat, n):
        """Concat-trick blend: unit-norm concat of sqrt(1-a)*fhat and
        sqrt(a)*khat, so downstream cosines equal (1-a)*cos_f + a*cos_k.
        Keys = last attention block's k projection (qk-space grouping)."""
        qkv = self._qkv_out.float()[0]                      # (prefix+N, 3d)
        self._qkv_out = None
        if qkv.shape[0] != n:
            qkv = qkv[-n:]
        d = qkv.shape[-1] // 3
        khat = F.normalize(qkv[:, d:2 * d], dim=-1)
        fhat = F.normalize(feat, dim=-1)
        a = self.attn_mix
        return torch.cat([(1 - a) ** 0.5 * fhat, a ** 0.5 * khat], dim=-1)

    @torch.no_grad()
    def _region_probs(self, pil_image, img_size, k):
        """One resolution pass: preprocess -> tokens -> region probs
        (gram seeds, or spherical k-means when cluster='kmeans').
        Returns (probs (K,N), feat (N,d), grid)."""
        x, grid = self._preprocess(pil_image, img_size)
        feat = self._tokens(x.to(self.device), grid)        # (N, d) fp32
        if self.cluster == "kmeans":
            probs = spherical_kmeans_probs(feat, k, self.kmeans_iters, self.seed)
        else:
            probs = gram_seed_probs(feat, k, self.tau, self.refine_rounds)
        return probs, feat, grid

    def _slic_labels(self, pil_image, size):
        """Pure-image baseline: SLIC superpixels in RGB, n_segments=K, no
        features. Runs at the same img_size scale as the feature path, then
        nearest-upsamples the label map to the original resolution."""
        try:
            from skimage.segmentation import slic
        except ImportError as e:
            raise ImportError(
                "cluster='slic' needs scikit-image, which is missing in this "
                "env. Install it with:\n  pip install scikit-image "
                "-i https://pypi.tuna.tsinghua.edu.cn/simple") from e
        H, W = size
        scale = float(self.img_size) / max(H, W)
        nh = max(1, int(round(H * scale)))
        nw = max(1, int(round(W * scale)))
        img = np.asarray(pil_image.resize((nw, nh), Image.BILINEAR))
        seg = slic(img, n_segments=max(2, self.k), start_label=1)
        labels = torch.from_numpy(np.ascontiguousarray(seg)).long()
        labels = F.interpolate(labels.view(1, 1, nh, nw).float(), size=(H, W),
                               mode="nearest")[0, 0].long()
        return labels.to(self.device)

    @torch.no_grad()
    def generate(self, pil_image):
        pil_image = pil_image.convert("RGB")
        H, W = pil_image.height, pil_image.width
        if self.cluster == "slic":
            return compact_labels(self._slic_labels(pil_image, (H, W))).to(torch.int32)
        probs, feat, grid = self._region_probs(pil_image, self.img_size, self.k)
        gh, gw = grid

        if self.multiscale:
            probs_c, _, grid_c = self._region_probs(
                pil_image, self.coarse_size, self.coarse_k)
            probs = merge_probs_multiscale(probs, feat, grid, probs_c, grid_c,
                                           self.merge_thresh)

        if self.pamr_iters > 0:
            labels = self._decide_pamr(probs, grid, (H, W), pil_image)
        elif self.upsample == "soft":
            labels = self._decide_fullres(probs, grid, (H, W))
        else:
            labels = self._decide_patchres(probs)           # (N,)
            labels = labels.view(1, 1, gh, gw).float()
            labels = F.interpolate(labels, size=(H, W), mode="nearest")
            labels = labels[0, 0].long()

        return compact_labels(labels).to(torch.int32)

    def _decide_patchres(self, probs):
        if self.binarize == "hard":
            return binarize_hard(probs, self.conf_thresh)
        member = topp_membership(probs, self.top_p)
        if self.conf_thresh > 0:                            # drop weak tokens
            member = member & (probs.max(0).values >= self.conf_thresh).unsqueeze(0)
        return paint_by_area(member)

    def _decide_fullres(self, probs, grid, size):
        gh, gw = grid
        if self.binarize == "hard":
            conf, arg = upsample_probs_argmax(probs, grid, size)
            labels = arg + 1
            if self.conf_thresh > 0:
                labels = labels.masked_fill(conf < self.conf_thresh, 0)
            return labels
        # topp at full res: nucleus decided at patch level (needs the full
        # per-token distribution), then each kept region's binary mask is
        # bilinearly upsampled and re-thresholded at 0.5 for smooth borders.
        member = topp_membership(probs, self.top_p)
        if self.conf_thresh > 0:
            member = member & (probs.max(0).values >= self.conf_thresh).unsqueeze(0)
        areas = member.sum(1)
        ids = torch.nonzero(areas > 0).squeeze(1)
        ids = ids[torch.argsort(areas[ids])]                # ascending area
        labels = torch.zeros(size, dtype=torch.long, device=probs.device)
        for new_id, k_idx in enumerate(ids.tolist(), start=1):
            m = member[k_idx].view(1, 1, gh, gw).float()
            m = F.interpolate(m, size=size, mode="bilinear", align_corners=False)
            labels[m[0, 0] > 0.5] = new_id
        return labels

    def _decide_pamr(self, probs, grid, size, pil_image):
        """Bilinearly upsample probs to (capped) pixel resolution, refine with
        PAMR against the RGB image, then take the hard/topp decision there.
        pamr_res == 0 -> native resolution (memory bounded by K-chunking)."""
        H, W = size
        if self.pamr_res and max(size) > self.pamr_res:
            s = float(self.pamr_res) / max(size)
            th, tw = max(1, int(round(H * s))), max(1, int(round(W * s)))
        else:
            th, tw = H, W
        img = pil_image.resize((tw, th), Image.BILINEAR)
        img = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)
        img = img.permute(2, 0, 1).unsqueeze(0).to(self.device)  # (1,3,th,tw)

        gh, gw = grid
        k = probs.shape[0]
        up = torch.empty(1, k, th, tw, device=probs.device)
        for s0 in range(0, k, 32):                          # chunked upsample
            p = probs[s0:s0 + 32].view(1, -1, gh, gw)
            up[0, s0:s0 + 32] = F.interpolate(
                p, size=(th, tw), mode="bilinear", align_corners=False)[0]
        up = pamr(img, up, iters=self.pamr_iters,
                  dilations=self.pamr_dilations)[0]         # (K, th, tw)

        if self.binarize == "hard":
            conf, arg = up.max(0)
            labels = arg + 1
            if self.conf_thresh > 0:
                labels = labels.masked_fill(conf < self.conf_thresh, 0)
        else:                                               # topp at pixel level
            flat = up.reshape(k, th * tw)
            flat = flat / flat.sum(0, keepdim=True).clamp(min=1e-6)
            member = topp_membership(flat, self.top_p)
            if self.conf_thresh > 0:
                member = member & (flat.max(0).values
                                   >= self.conf_thresh).unsqueeze(0)
            labels = paint_by_area(member).view(th, tw)

        labels = labels.view(th, tw)
        if (th, tw) != (H, W):
            labels = F.interpolate(labels.view(1, 1, th, tw).float(),
                                   size=(H, W), mode="nearest")[0, 0].long()
        return labels
