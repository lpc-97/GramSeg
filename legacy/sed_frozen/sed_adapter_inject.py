"""Monkey-patch: optionally insert the Step-2 adapter in front of SED's cost map.

Importing this module wraps Aggregator.forward
(sed/modeling/transformer/model.py) so that, iff SED_ADAPTER_PATH is set,
img_feats (clip_vis_dense, B x 640 x 24 x 24) is passed through the adapter
before `corr = self.correlation(img_feats, ...)`. img_feats has no other
consumer inside Aggregator.forward, so this is the complete injection.

When SED_ADAPTER_PATH is unset/empty the original forward is called with
untouched arguments -> bit-exact to vanilla SED.

Requires the SED repo root on sys.path (import after cd-ing there, or via
train_net_adapter.py).
"""
import os

import torch

from sed.modeling.transformer.model import Aggregator
from adapter_module import Adapter

_orig_forward = Aggregator.forward
_ATTR = "_step2_adapter"


def _forward(self, img_feats, text_feats, appearance_guidance, logit_scale):
    path = os.environ.get("SED_ADAPTER_PATH", "")
    if path:
        ad = getattr(self, _ATTR, None)
        if ad is None:
            ad = Adapter.from_checkpoint(path)
            ad = ad.to(device=img_feats.device, dtype=torch.float32).eval()
            for p in ad.parameters():
                p.requires_grad_(False)
            # bypass nn.Module registration: keep the adapter out of
            # state_dict()/named_parameters() of the evaluated model
            object.__setattr__(self, _ATTR, ad)
            print(f"[sed-adapter] active: {path} "
                  f"(d={ad.d}, hidden={ad.hidden}, gate={ad.gate.item():.3f})")
        with torch.no_grad():
            img_feats = ad.forward_map(img_feats.float())
    return _orig_forward(self, img_feats, text_feats, appearance_guidance,
                         logit_scale)


Aggregator.forward = _forward
print("[sed-adapter] Aggregator.forward hook installed "
      f"(SED_ADAPTER_PATH={'set' if os.environ.get('SED_ADAPTER_PATH') else 'unset -> no-op'})")
