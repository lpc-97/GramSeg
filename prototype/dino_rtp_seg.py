"""DINO-RTP-Seg: Real-Time Promptable Open-Vocabulary Segmentation on frozen DINOv3.

Minimal prototype implementing the four contributions from METHOD_SPEC.md:
  C1  single frozen DINOv3 backbone, features reused for masks AND semantics
  C2  Gram-Seed mask proposal (FPS seeds + one-step affinity propagation)
  C3  prompt-conditioned sparse routing (text / exemplar / point / box -> top-m regions)
  C4  presence gating for early exit

Backbone: timm DINOv3 (vit_small/base_patch16_dinov3.lvd1689m), loaded offline
from a local safetensors file. Text tower: SigLIP (server-cached) via adapter.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Backbone wrapper (frozen DINOv3 via timm, offline weights)
# --------------------------------------------------------------------------- #

class FrozenDINOv3(nn.Module):
    """Frozen DINOv3 producing L2-normalizable patch tokens + two shallow taps."""

    TIMM_NAMES = {
        "vits16": "vit_small_patch16_dinov3.lvd1689m",
        "vitb16": "vit_base_patch16_dinov3.lvd1689m",
    }

    def __init__(self, variant="vits16", weights_file=None, img_size=640):
        super().__init__()
        import timm
        kwargs = dict(pretrained=weights_file is not None, num_classes=0,
                      img_size=img_size, dynamic_img_size=True)
        if weights_file is not None:
            kwargs["pretrained_cfg_overlay"] = dict(file=weights_file)
        self.model = timm.create_model(self.TIMM_NAMES[variant], **kwargs)
        self.model.eval().requires_grad_(False)
        self.embed_dim = self.model.embed_dim
        self.patch = 16
        self.num_prefix = self.model.num_prefix_tokens  # cls + register tokens
        # taps for the refinement FPN: ~1/3 and ~2/3 depth
        depth = len(self.model.blocks)
        self.tap_ids = {depth // 3 - 1, 2 * depth // 3 - 1}

    @torch.no_grad()
    def forward(self, x):
        """Returns final patch tokens (B,N,d), shallow taps [(B,N,d)], grid (h,w)."""
        h, w = x.shape[-2] // self.patch, x.shape[-1] // self.patch
        final, taps = self.model.forward_intermediates(
            x, indices=sorted(self.tap_ids), return_prefix_tokens=False,
            norm=True, output_fmt="NLC", intermediates_only=False)
        if final.shape[1] != h * w:                # strip cls/register tokens
            final = final[:, -h * w:]
        taps = [t if t.shape[1] == h * w else t[:, -h * w:] for t in taps]
        return final, taps, (h, w)


# --------------------------------------------------------------------------- #
# C2: Gram-Seed mask proposal
# --------------------------------------------------------------------------- #

def farthest_point_sample(feat, k):
    """feat: (B,N,d) L2-normalized. Returns (B,k) seed indices. O(kN)."""
    B, N, _ = feat.shape
    idx = torch.zeros(B, k, dtype=torch.long, device=feat.device)
    center = F.normalize(feat.mean(1, keepdim=True), dim=-1)
    dist = 1.0 - (feat @ center.transpose(1, 2)).squeeze(-1)  # (B,N)
    idx[:, 0] = dist.argmax(-1)
    min_d = torch.full((B, N), 1e9, device=feat.device)
    for i in range(1, k):
        last = feat.gather(1, idx[:, i - 1, None, None].expand(-1, 1, feat.shape[-1]))
        d = 1.0 - (feat @ last.transpose(1, 2)).squeeze(-1)
        min_d = torch.minimum(min_d, d)
        idx[:, i] = min_d.argmax(-1)
    return idx


class TinyFormer(nn.Module):
    """2-layer self-attention over the K region embeddings only."""

    def __init__(self, d, heads=8, layers=2):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d, heads, d * 2, dropout=0.0,
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)

    def forward(self, r):
        return self.enc(r)


class GramSeedProposal(nn.Module):
    """FPS seeds -> one-step affinity propagation -> K soft masks + embeddings."""

    def __init__(self, d, k=64, tau=0.07, refine_rounds=1):
        super().__init__()
        self.k, self.tau, self.refine_rounds = k, tau, refine_rounds
        self.tiny = TinyFormer(d)

    def forward(self, feat):
        """feat: (B,N,d). Returns masks (B,K,N), regions (B,K,d)."""
        fhat = F.normalize(feat, dim=-1)
        seeds = farthest_point_sample(fhat, self.k)                    # (B,K)
        q = fhat.gather(1, seeds[..., None].expand(-1, -1, feat.shape[-1]))
        masks = torch.softmax(q @ fhat.transpose(1, 2) / self.tau, dim=1)  # (B,K,N)
        regions = (masks @ feat) / masks.sum(-1, keepdim=True).clamp(min=1e-6)
        regions = self.tiny(regions)
        for _ in range(self.refine_rounds):
            rhat = F.normalize(regions, dim=-1)
            masks = torch.softmax(rhat @ fhat.transpose(1, 2) / self.tau, dim=1)
            regions = (masks @ feat) / masks.sum(-1, keepdim=True).clamp(min=1e-6)
        return masks, regions


# --------------------------------------------------------------------------- #
# C3: prompt encoders + sparse router
# --------------------------------------------------------------------------- #

class PromptEncoder(nn.Module):
    """Maps text / exemplar / point / box prompts into the region embedding space."""

    def __init__(self, d, d_text):
        super().__init__()
        self.text_proj = nn.Linear(d_text, d)
        self.visual_mlp = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.pos_mlp = nn.Sequential(nn.Linear(4, d), nn.GELU(), nn.Linear(d, d))

    def encode_text(self, text_emb):                       # (P,d_text)
        return self.text_proj(text_emb)

    def encode_exemplar(self, feat, grid, boxes):
        """feat (B,N,d); boxes (P,4) normalized xyxy -> mask-pool + MLP."""
        h, w = grid
        fmap = feat.transpose(1, 2).reshape(feat.shape[0], -1, h, w)
        out = []
        for b in boxes:
            x0, y0, x1, y1 = (b * torch.tensor([w, h, w, h], device=feat.device))
            x0, y0 = int(x0.clamp(0, w - 1)), int(y0.clamp(0, h - 1))
            x1, y1 = int(x1.clamp(x0 + 1, w)), int(y1.clamp(y0 + 1, h))
            out.append(fmap[0, :, y0:y1, x0:x1].mean(dim=(1, 2)))
        return self.visual_mlp(torch.stack(out))

    def encode_point(self, feat, grid, points):
        """points (P,2) normalized xy -> patch feature + positional MLP."""
        h, w = grid
        ij = (points * torch.tensor([w, h], device=feat.device)).long()
        flat = (ij[:, 1].clamp(0, h - 1) * w + ij[:, 0].clamp(0, w - 1))
        pf = feat[0, flat]
        pos = torch.cat([points, points], dim=-1)
        return self.visual_mlp(pf) + self.pos_mlp(pos)


class SparseRouter(nn.Module):
    """C3+C4: route each prompt to top-m regions, with presence gating."""

    def __init__(self, d, d_text, m=8):
        super().__init__()
        self.m = m
        self.wq, self.wk = nn.Linear(d, d), nn.Linear(d, d)
        self.to_text = nn.Linear(d, d_text)               # psi_R: region -> text space
        self.presence = nn.Sequential(nn.Linear(d + 1, d // 2), nn.GELU(),
                                      nn.Linear(d // 2, 1))
        self.scale = d ** -0.5

    def forward(self, prompts, regions, masks):
        """prompts (P,d); regions (B,K,d); masks (B,K,N).
        Returns per-prompt masks (P,N), agg embeddings (P,d), presence (P,)."""
        q = self.wq(prompts)                               # (P,d)
        kk = self.wk(regions[0])                           # (K,d)
        score = q @ kk.t() * self.scale                    # (P,K)
        top, idx = score.topk(self.m, dim=-1)              # (P,m)
        alpha = torch.softmax(top, dim=-1).to(score.dtype)
        # sparse routing weights (P,K): zero outside top-m
        w = torch.zeros_like(score).scatter_(1, idx, alpha)
        ymask = w @ masks[0]                               # (P,N)
        z = w @ regions[0]                                 # (P,d)
        pres = self.presence(torch.cat([prompts, score.max(-1, keepdim=True).values], -1))
        return ymask, z, pres.squeeze(-1), w


# --------------------------------------------------------------------------- #
# Mask refinement head (patch-res -> 1/4 res)
# --------------------------------------------------------------------------- #

class RefineHead(nn.Module):
    """Lightweight FPN guided by two shallow backbone taps; <2M params."""

    def __init__(self, d, hidden=64):
        super().__init__()
        self.lat = nn.ModuleList([nn.Conv2d(d, hidden, 1) for _ in range(2)])
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden + 1, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, 1, 1))

    def forward(self, mask_patch, taps, grid):
        """mask_patch (P,N); taps [(B,N,d)x2]; -> (P, 4h, 4w) logits."""
        h, w = grid
        m = mask_patch.reshape(-1, 1, h, w)
        guide = 0
        for tap, lat in zip(taps, self.lat):
            g = lat(tap.transpose(1, 2).reshape(1, -1, h, w))
            guide = guide + g
        g = guide
        for _ in range(2):                                 # 2x twice -> 1/4 res
            m = F.interpolate(m, scale_factor=2, mode="bilinear")
            g = F.interpolate(guide, size=m.shape[-2:], mode="bilinear")
            m = m + self.fuse(torch.cat([m, g.expand(m.shape[0], -1, -1, -1)], 1))
        return m.squeeze(1), g[0]                          # masks (K,H,W), guide (hidden,H,W)


# --------------------------------------------------------------------------- #
# Full model
# --------------------------------------------------------------------------- #

class DinoRTPSeg(nn.Module):
    def __init__(self, variant="vits16", weights_file=None, img_size=640,
                 k=64, m=8, d_text=1152, use_kernel=False):
        super().__init__()
        self.backbone = FrozenDINOv3(variant, weights_file, img_size)
        d = self.backbone.embed_dim
        self.proposal = GramSeedProposal(d, k=k)
        self.prompt_enc = PromptEncoder(d, d_text)
        self.router = SparseRouter(d, d_text, m=m)
        self.refine = RefineHead(d)
        self.use_kernel = use_kernel
        # dynamic mask kernel: per-prompt pixel-level correction on top of the
        # region composition (breaks the linear-combination expressivity cap)
        self.mask_kernel = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 64))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))

    def compose_hi(self, w, z, reg_hi, guide):
        """Per-prompt hi-res masks: sparse region combo (+ dynamic kernel)."""
        base = (w @ reg_hi.flatten(1)).reshape(-1, *reg_hi.shape[-2:])
        if not self.use_kernel:
            return base
        kern = self.mask_kernel(z)                        # (P,64)
        corr = torch.einsum("pc,chw->phw", kern, guide)
        return base + corr

    def trainable_parameters(self):
        return [p for n, p in self.named_parameters()
                if not n.startswith("backbone.") and p.requires_grad]

    def forward(self, image, text_emb=None, points=None, boxes=None,
                exemplar_boxes=None, presence_thresh=None, refine=True):
        feat, taps, grid = self.backbone(image)
        masks, regions = self.proposal(feat)

        prompts, kinds = [], []
        if text_emb is not None:
            prompts.append(self.prompt_enc.encode_text(text_emb)); kinds += ["text"] * len(text_emb)
        if exemplar_boxes is not None:
            prompts.append(self.prompt_enc.encode_exemplar(feat, grid, exemplar_boxes)); kinds += ["exemplar"] * len(exemplar_boxes)
        if points is not None:
            prompts.append(self.prompt_enc.encode_point(feat, grid, points)); kinds += ["point"] * len(points)
        if boxes is not None:
            prompts.append(self.prompt_enc.encode_exemplar(feat, grid, boxes)); kinds += ["box"] * len(boxes)
        if not prompts:
            raise ValueError("need at least one prompt")
        prompts = torch.cat(prompts, 0)

        ymask, z, pres, w = self.router(prompts, regions, masks)

        if presence_thresh is not None:                    # C4 early exit
            keep = torch.sigmoid(pres) >= presence_thresh
            ymask, z, w = ymask[keep], z[keep], w[keep]
            kinds = [k_ for k_, kp in zip(kinds, keep.tolist()) if kp]

        out = {"mask_patch": ymask, "embed": z, "presence": pres,
               "region_masks": masks, "regions": regions, "grid": grid,
               "kinds": kinds, "text_embed": self.router.to_text(z)}
        if refine and ymask.shape[0] > 0:
            # refine the K region masks ONCE (vocab-independent), then compose
            # per-prompt hi-res masks with the sparse routing weights.
            reg_hi, guide = self.refine(masks[0], taps, grid)   # (K,4h,4w), (64,4h,4w)
            out["region_masks_hi"] = reg_hi
            out["mask_hi"] = self.compose_hi(w, z, reg_hi, guide)
        return out

    @torch.no_grad()
    def semantic_segmentation(self, image, class_text_emb):
        """Open-vocab semantic seg: route every class, per-pixel argmax."""
        out = self.forward(image, text_emb=class_text_emb, refine=True)
        logits = out["mask_hi"]                            # (C, 4h, 4w)
        sims = F.normalize(out["text_embed"], dim=-1) @ F.normalize(class_text_emb, dim=-1).t()
        cls_score = sims.diag()[:, None, None] * self.logit_scale.exp()
        return (logits + cls_score).argmax(0), out
