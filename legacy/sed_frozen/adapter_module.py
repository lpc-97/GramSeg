"""Shared Adapter definition for SED Step-2 (cached-feature adapter).

Residual MLP + learnable gate operating per-token on the 640-d cost-map
image features (clip_vis_dense). The adapter output is *unnormalized*;
normalization is applied downstream:
  - at train time by the loss (F.normalize before the text dot-product),
  - at eval time by Aggregator.correlation (F.normalize(img_feats, dim=1)).
Since correlation renormalizes anyway, only the *direction* of the adapted
feature matters, which train/eval treat identically.
"""
import torch
import torch.nn as nn


class Adapter(nn.Module):
    """emb -> emb + gate * MLP(emb).  Params: 2*d*hidden + O(d)."""

    def __init__(self, d: int = 640, hidden: int = 3072):
        super().__init__()
        self.d = d
        self.hidden = hidden
        self.mlp = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, hidden),
            nn.GELU(),
            nn.Linear(hidden, d),
        )
        # small gate: near-identity at init, trains quickly off cached feats
        self.gate = nn.Parameter(torch.tensor(0.1))
        # temperature used only by the training loss (harmless extra scalar
        # in the checkpoint; ignored at eval where correlation has its own).
        self.logit_scale = nn.Parameter(torch.tensor(4.0))
        # zero-init the last linear so the residual starts exactly at 0
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        """e: (..., d) token-space features. Returns unnormalized adapted."""
        return e + self.gate * self.mlp(e)

    def forward_map(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) feature map (SED clip_vis_dense layout)."""
        e = x.permute(0, 2, 3, 1)          # B H W C  (LayerNorm over C)
        e = self.forward(e)
        return e.permute(0, 3, 1, 2).contiguous()

    @staticmethod
    def from_checkpoint(path: str) -> "Adapter":
        ck = torch.load(path, map_location="cpu")
        sd = ck.get("state_dict", ck)
        hidden, d = sd["mlp.1.weight"].shape   # (hidden, d)
        ad = Adapter(d=d, hidden=hidden)
        ad.load_state_dict(sd)
        return ad
