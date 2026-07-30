#!/usr/bin/env python
"""Drop-in replacement for SED's train_net.py with optional cost-map adapter.

Usage (from the SED repo root, same CLI as train_net.py):
  SED_ADAPTER_PATH=/path/adapter.pth python step2/train_net_adapter.py \
      --config configs/convnextB_768.yaml --num-gpus 1 --eval-only ...

With SED_ADAPTER_PATH unset the run is bit-exact to `python train_net.py ...`
(the hook forwards untouched arguments). Single-GPU only: with --num-gpus 1
detectron2's launch() runs in-process, so the monkey-patch installed here is
guaranteed to be live in the worker.
"""
import os
import runpy
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_repo = os.getcwd()                       # must be the SED repo root
assert os.path.exists(os.path.join(_repo, "train_net.py")), \
    "run from the SED repo root (train_net.py not found in cwd)"
sys.path.insert(0, _here)                 # adapter_module, sed_adapter_inject
sys.path.insert(0, _repo)                 # sed package

import sed_adapter_inject  # noqa: F401,E402  (installs the hook)

runpy.run_path(os.path.join(_repo, "train_net.py"), run_name="__main__")
