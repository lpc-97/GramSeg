"""COCO-Stuff 10k subset dataset: images/*.jpg + labels/*.png (raw ids).

Returns (x, y) with y in {0=ignore, 1..171} to match the ADE convention.
"""
import os, random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from train_ade import MEAN, STD
from coco_names import CLS_TO_TRAIN

_LUT = np.full(256, 0, dtype=np.int64)          # default ignore->0
for raw, tr in CLS_TO_TRAIN.items():
    if raw != 255:
        _LUT[raw] = tr + 1                       # shift to 1..171


class CocoStuff10k(Dataset):
    def __init__(self, root, size=640):
        self.img_dir = os.path.join(root, "images")
        self.lab_dir = os.path.join(root, "labels")
        self.files = sorted(os.listdir(self.img_dir))
        self.size = size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        fn = self.files[i]
        img = Image.open(os.path.join(self.img_dir, fn)).convert("RGB")
        ann = Image.open(os.path.join(self.lab_dir, fn.replace(".jpg", ".png")))
        s = self.size
        w, h = img.size
        scale = random.uniform(0.75, 1.5) * s / min(w, h)
        nw, nh = max(s, int(w * scale)), max(s, int(h * scale))
        img = img.resize((nw, nh), Image.BILINEAR)
        ann = ann.resize((nw, nh), Image.NEAREST)
        x0, y0 = random.randint(0, nw - s), random.randint(0, nh - s)
        img = img.crop((x0, y0, x0 + s, y0 + s))
        ann = ann.crop((x0, y0, x0 + s, y0 + s))
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            ann = ann.transpose(Image.FLIP_LEFT_RIGHT)
        x = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        x = (x - MEAN) / STD
        y = torch.from_numpy(_LUT[np.array(ann)])
        return x, y
