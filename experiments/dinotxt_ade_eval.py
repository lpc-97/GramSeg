#!/usr/bin/env python
# dino.txt (DINOv3 ViT-L/16 + dinotxt head) zero-shot ADE20K 评测
# 协议 A --mode official : 官方 notebook 配方(短边512, slide 384/192, softmax 累积)→ 对论文数校准
# 协议 B --mode ours     : 我们全线协议(mmseg Resize (2048,448) keep-ratio, whole)→ 论文 baseline 行
# 文本:80 CLIP 模板,encode_text 取后半段(patch 对齐半),L2 归一→均值→归一(官方 notebook 逐位一致)
import argparse, math, os, sys, glob
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.expanduser("~/dinov3_repo"))
from dinov3.hub.dinotxt import dinov3_vitl16_dinotxt_tet1280d20h24l

ADE_ROOT = os.path.expanduser("~/dino_rtp_seg/data/ADEChallengeData2016")
BB = None   # filled by args
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

PROMPT_TEMPLATES = (
    "a bad photo of a {0}.", "a photo of many {0}.", "a sculpture of a {0}.",
    "a photo of the hard to see {0}.", "a low resolution photo of the {0}.",
    "a rendering of a {0}.", "graffiti of a {0}.", "a bad photo of the {0}.",
    "a cropped photo of the {0}.", "a tattoo of a {0}.", "the embroidered {0}.",
    "a photo of a hard to see {0}.", "a bright photo of a {0}.", "a photo of a clean {0}.",
    "a photo of a dirty {0}.", "a dark photo of the {0}.", "a drawing of a {0}.",
    "a photo of my {0}.", "the plastic {0}.", "a photo of the cool {0}.",
    "a close-up photo of a {0}.", "a black and white photo of the {0}.",
    "a painting of the {0}.", "a painting of a {0}.", "a pixelated photo of the {0}.",
    "a sculpture of the {0}.", "a bright photo of the {0}.", "a cropped photo of a {0}.",
    "a plastic {0}.", "a photo of the dirty {0}.", "a jpeg corrupted photo of a {0}.",
    "a blurry photo of the {0}.", "a photo of the {0}.", "a good photo of the {0}.",
    "a rendering of the {0}.", "a {0} in a video game.", "a photo of one {0}.",
    "a doodle of a {0}.", "a close-up photo of the {0}.", "a photo of a {0}.",
    "the origami {0}.", "the {0} in a video game.", "a sketch of a {0}.",
    "a doodle of the {0}.", "a origami {0}.", "a low resolution photo of a {0}.",
    "the toy {0}.", "a rendition of the {0}.", "a photo of the clean {0}.",
    "a photo of a large {0}.", "a rendition of a {0}.", "a photo of a nice {0}.",
    "a photo of a weird {0}.", "a blurry photo of a {0}.", "a cartoon {0}.",
    "art of a {0}.", "a sketch of the {0}.", "a embroidered {0}.",
    "a pixelated photo of a {0}.", "itap of the {0}.",
    "a jpeg corrupted photo of the {0}.", "a good photo of a {0}.",
    "a plushie {0}.", "a photo of the nice {0}.", "a photo of the small {0}.",
    "a photo of the weird {0}.", "the cartoon {0}.", "art of the {0}.",
    "a drawing of the {0}.", "a photo of the large {0}.",
    "a black and white photo of a {0}.", "the plushie {0}.", "a dark photo of a {0}.",
    "itap of a {0}.", "graffiti of the {0}.", "a toy {0}.", "itap of my {0}.",
    "a photo of a cool {0}.", "a photo of a small {0}.", "a tattoo of the {0}.",
)


def load_model(args):
    model, tokenizer = dinov3_vitl16_dinotxt_tet1280d20h24l(
        weights=args.head_weights,
        backbone_weights=args.backbone_weights,
        bpe_path_or_url="file://" + args.bpe,
    )
    model = model.cuda().eval()
    return model, tokenizer


@torch.no_grad()
def build_text_features(model, tokenizer, class_names):
    feats_all = []
    for name in class_names:
        text = [t.format(name) for t in PROMPT_TEMPLATES]
        tokens = tokenizer.tokenize(text).cuda()
        feats = model.encode_text(tokens)          # [P, 2048]
        feats = feats[:, feats.shape[1] // 2:]     # patch-aligned half [P, 1024]
        feats = F.normalize(feats, p=2, dim=-1).mean(dim=0)
        feats = F.normalize(feats, p=2, dim=0)
        feats_all.append(feats)
    return torch.stack(feats_all)                  # [C, 1024]


@torch.no_grad()
def encode_patches(model, img):                    # img [3,H,W] normalized
    P = model.visual_model.backbone.patch_size
    _, H, W = img.shape
    nH, nW = math.ceil(H / P) * P, math.ceil(W / P) * P
    x = img.unsqueeze(0)
    if (H, W) != (nH, nW):
        x = F.interpolate(x, size=(nH, nW), mode="bicubic", align_corners=False)
    # 官方 notebook 解包:cls, _, patch —— 密集预测用第三个返回值(backbone patch tokens),
    # 不是第二个(head 投影 patch tokens)。此前取错导致 ADE −5/City −13 的校准差。
    _, _, patch_tokens = model.visual_model.get_class_and_patch_tokens(x)
    feats = patch_tokens.reshape(1, nH // P, nW // P, -1).squeeze(0)  # [h,w,D]
    return F.normalize(feats, p=2, dim=-1)


@torch.no_grad()
def predict_whole(model, img, text_feats):
    feats = encode_patches(model, img)                       # [h,w,D]
    return torch.einsum("cd,hwd->chw", text_feats, feats)    # [C,h,w]


@torch.no_grad()
def predict_slide(model, img, text_feats, side, stride):
    _, H, W = img.shape
    C = text_feats.shape[0]
    probs = torch.zeros(C, H, W, device="cuda")
    counts = torch.zeros(H, W, device="cuda")
    hg = max(H - side + stride - 1, 0) // stride + 1
    wg = max(W - side + stride - 1, 0) // stride + 1
    for i in range(hg):
        for j in range(wg):
            y1, x1 = i * stride, j * stride
            y2, x2 = min(y1 + side, H), min(x1 + side, W)
            y1, x1 = max(y2 - side, 0), max(x2 - side, 0)
            cos = predict_whole(model, img[:, y1:y2, x1:x2], text_feats)
            cos = F.interpolate(cos.unsqueeze(0), size=(y2 - y1, x2 - x1),
                                mode="bilinear", align_corners=False).squeeze(0)
            probs[:, y1:y2, x1:x2] += cos.softmax(dim=0)
            counts[y1:y2, x1:x2] += 1
    return probs / counts


def resize_short(img, size):                       # PIL, bicubic
    w, h = img.size
    if w < h:
        nw, nh = size, int(size * h / w)
    else:
        nh, nw = size, int(size * w / h)
    return img.resize((nw, nh), Image.BICUBIC)


def resize_mmseg(img, scale_long=2048, scale_short=448):
    # mmseg Resize(scale=(2048,448), keep_ratio=True):缩放比 = min(2048/max, 448/min)
    w, h = img.size
    r = min(scale_long / max(w, h), scale_short / min(w, h))
    return img.resize((int(w * r + 0.5), int(h * r + 0.5)), Image.BICUBIC)


CITY_NAMES = ["road", "sidewalk", "building", "wall", "fence", "pole",
              "traffic light", "traffic sign", "vegetation", "terrain", "sky",
              "person", "rider", "car", "truck", "bus", "train", "motorcycle",
              "bicycle"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["official", "ours"], required=True)
    ap.add_argument("--dataset", choices=["ade", "city"], default="ade")
    ap.add_argument("--resize", type=int, default=512)
    ap.add_argument("--head-weights", required=True)
    ap.add_argument("--backbone-weights", required=True)
    ap.add_argument("--bpe", required=True)
    ap.add_argument("--names", default=os.path.expanduser("~/CorrCLIP/configs/cls_ade20k.txt"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if args.dataset == "ade":
        names = [l.strip() for l in open(args.names) if l.strip()]
        assert len(names) == 150, f"expect 150 ADE classes, got {len(names)}"
        img_dir = os.path.join(ADE_ROOT, "images/validation")
        ann_dir = os.path.join(ADE_ROOT, "annotations/validation")
        files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
        ann_of = lambda fp: os.path.join(ann_dir, os.path.basename(fp).replace(".jpg", ".png"))
        reduce_zero = True
    else:
        names = CITY_NAMES
        city_root = os.path.expanduser("~/CorrCLIP/data/cityscapes")
        files = sorted(glob.glob(os.path.join(city_root, "leftImg8bit/val/*/*_leftImg8bit.png")))
        ann_map = {os.path.basename(p).replace("_gtFine_labelTrainIds.png", ""): p
                   for p in glob.glob(os.path.join(city_root, "gtFine/val/*/*_gtFine_labelTrainIds.png"))}
        ann_of = lambda fp: ann_map[os.path.basename(fp).replace("_leftImg8bit.png", "")]
        reduce_zero = False

    model, tokenizer = load_model(args)
    text_feats = build_text_features(model, tokenizer, names)
    print(f"text_feats {tuple(text_feats.shape)} dataset={args.dataset} resize={args.resize}", flush=True)

    if args.limit:
        files = files[: args.limit]
    C = len(names)
    inter = torch.zeros(C, dtype=torch.float64)
    union = torch.zeros(C, dtype=torch.float64)

    for idx, fp in enumerate(files):
        pil = Image.open(fp).convert("RGB")
        gt = np.array(Image.open(ann_of(fp)))
        gt = torch.from_numpy(gt.astype(np.int64)).cuda()
        if reduce_zero:
            gt = torch.where((gt == 0) | (gt == 255), 255, gt - 1)   # reduce zero label

        pil_r = resize_short(pil, args.resize) if args.mode == "official" else resize_mmseg(pil)
        img = torch.from_numpy(np.array(pil_r)).permute(2, 0, 1).float() / 255.0
        img = ((img - IMAGENET_MEAN) / IMAGENET_STD).cuda()

        if args.mode == "official":
            pred = predict_slide(model, img, text_feats, side=384, stride=192)
        else:
            pred = predict_whole(model, img, text_feats)

        pred = F.interpolate(pred.unsqueeze(0), size=gt.shape, mode="bilinear",
                             align_corners=False).squeeze(0).argmax(dim=0)

        valid = gt != 255
        p, g = pred[valid], gt[valid]
        for c in torch.unique(torch.cat([p, g])):
            pi, gi = p == c, g == c
            inter[c] += (pi & gi).sum().item()
            union[c] += (pi | gi).sum().item()
        if (idx + 1) % 200 == 0:
            iou = (inter / union.clamp(min=1)).numpy()
            print(f"[{idx+1}/{len(files)}] running mIoU={100*np.nanmean(np.where(union.numpy()>0, iou, np.nan)):.2f}", flush=True)

    iou = (inter / union.clamp(min=1)).numpy()
    miou = 100 * np.nanmean(np.where(union.numpy() > 0, iou, np.nan))
    print(f"FINAL mode={args.mode} dataset={args.dataset} resize={args.resize} "
          f"images={len(files)} mIoU={miou:.2f}", flush=True)


if __name__ == "__main__":
    main()
