"""Zero-shot region classification via SigLIP mask attention pooling.

FC-CLIP-style: restrict SigLIP's attention-pooling head (MAP) to each region's
patches, producing text-aligned region embeddings WITHOUT any training.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

SIGLIP_MEAN = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
SIGLIP_STD = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMNET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class SigLIPRegionClassifier(nn.Module):
    """Frozen SigLIP vision tower + per-region mask attention pooling."""

    def __init__(self, vision_dir, res=384):
        super().__init__()
        from transformers import SiglipVisionModel
        self.vision = SiglipVisionModel.from_pretrained(vision_dir).eval()
        self.vision.requires_grad_(False)
        self.res = res
        self.grid = res // self.vision.config.patch_size          # 27 for so400m/14

    @torch.no_grad()
    def imagenet_to_siglip(self, x):
        """x: (B,3,H,W) ImageNet-normalized -> SigLIP view at self.res."""
        dev = x.device
        raw = x * IMNET_STD.to(dev) + IMNET_MEAN.to(dev)
        raw = F.interpolate(raw, size=(self.res, self.res), mode="bilinear",
                            align_corners=False)
        return (raw - SIGLIP_MEAN.to(dev)) / SIGLIP_STD.to(dev)

    @torch.no_grad()
    def region_embeddings(self, x_imnet, region_masks, mask_grid):
        """x_imnet: (1,3,H,W); region_masks: (K,N) soft over DINOv3 patch grid
        mask_grid: (h,w) of that grid. Returns (K, d) text-aligned embeddings."""
        vm = self.vision.vision_model
        px = self.imagenet_to_siglip(x_imnet)
        hidden = vm.encoder(vm.embeddings(
            px, interpolate_pos_encoding=(self.res != 384))).last_hidden_state  # (1,P,d)
        hidden = vm.post_layernorm(hidden)
        P = hidden.shape[1]
        K = region_masks.shape[0]

        # resize region masks to SigLIP patch grid
        h, w = mask_grid
        rm = region_masks.reshape(K, 1, h, w)
        rm = F.interpolate(rm, size=(self.grid, self.grid), mode="bilinear",
                           align_corners=False).reshape(K, P)      # (K,P)
        hard = rm >= 0.6 * rm.amax(dim=-1, keepdim=True)           # per-region binarize
        hard = hard | (rm >= rm.amax(dim=-1, keepdim=True) * 0.999)  # keep peak

        # mask attention pooling with the MAP head, batched over K
        head = vm.head
        probe = head.probe.expand(K, -1, -1)                       # (K,1,d)
        keys = hidden.expand(K, -1, -1)                            # (K,P,d)
        attn_out, _ = head.attention(probe, keys, keys,
                                     key_padding_mask=~hard)       # (K,1,d)
        res = attn_out
        res = res + head.mlp(head.layernorm(res))
        return F.normalize(res[:, 0].float(), dim=-1)              # (K,d)

    @torch.no_grad()
    def classify(self, x_imnet, region_masks, mask_grid, text_emb, scale=100.0):
        """Returns (K, C) class probabilities per region (zero-shot)."""
        emb = self.region_embeddings(x_imnet, region_masks, mask_grid)
        return torch.softmax(emb @ text_emb.t().float() * scale, dim=-1)
