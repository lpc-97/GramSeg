#!/usr/bin/env python
# GRAM_VOCAB_MEMFIX 补丁:10k 词表探针专用,类维分块上采样+argmax 直出标签,
# 避免 [Q,H,W] 全尺寸稠密 logits(通用 mmseg 接口的显存爆点)。env 不开时代码路径逐字节不变。
#
# Usage: python apply_memfix_patch.py [path/to/corrclip_segmentor.py]
#        (defaults to ~/CorrCLIP/corrclip_segmentor.py, the path used in our runs)
# Aborts with AssertionError if the file is already patched, or if either anchor
# is missing (i.e. the upstream commit is not the pinned one).
import os
import sys

P = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1
                       else "~/CorrCLIP/corrclip_segmentor.py")
src = open(P).read()
assert "GRAM_VOCAB_MEMFIX" not in src, "already patched"

old_slide = """        preds = preds / count_mat
        img_size = img_metas[0]['ori_shape'][:2]
        logits = nn.functional.interpolate(preds, size=img_size, mode='bilinear')"""
new_slide = """        img_size = img_metas[0]['ori_shape'][:2]
        if os.environ.get('GRAM_VOCAB_MEMFIX', '0') == '1':
            # label-out path: in-place normalize + class-chunked upsample/argmax.
            preds /= count_mat
            best_s, best_i, CH = None, None, 500
            for c0 in range(0, preds.shape[1], CH):
                chunk = nn.functional.interpolate(
                    preds[:, c0:c0 + CH], size=img_size, mode='bilinear')
                s, ix = chunk.max(dim=1, keepdim=True)
                if best_s is None:
                    best_s, best_i = s, ix + c0
                else:
                    m = s > best_s
                    best_s = torch.where(m, s, best_s)
                    best_i = torch.where(m, ix + c0, best_i)
            torch.cuda.empty_cache()
            return (best_s, best_i)
        preds = preds / count_mat
        logits = nn.functional.interpolate(preds, size=img_size, mode='bilinear')"""
assert src.count(old_slide) == 1, "slide anchor not found"
src = src.replace(old_slide, new_slide)

old_pred = """        seg_logits = self.forward_slide(inputs, instance_masks, batch_img_metas, self.slide_stride, self.slide_crop)

        return self.postprocess_result(seg_logits, data_samples)"""
new_pred = """        seg_logits = self.forward_slide(inputs, instance_masks, batch_img_metas, self.slide_stride, self.slide_crop)

        if isinstance(seg_logits, tuple):   # GRAM_VOCAB_MEMFIX label path
            assert max(self.query_idx) + 1 == len(self.query_idx), \\
                'memfix path assumes 1:1 query:class (no synonyms/background split)'
            best_s, best_i = seg_logits
            for i in range(best_i.shape[0]):
                seg_pred = best_i[i]
                mask_values = torch.unique(self.instance_masks[i])
                mask_values = mask_values[1:]
                for mv in mask_values:
                    m = (self.instance_masks[i] == mv).unsqueeze(0)
                    seg_pred[m] = torch.mode(seg_pred[m])[0]
                data_samples[i].set_data({
                    'seg_logit': PixelData(**{'data': best_s[i]}),
                    'pred_sem_seg': PixelData(**{'data': seg_pred})
                })
            return data_samples

        return self.postprocess_result(seg_logits, data_samples)"""
assert src.count(old_pred) == 1, "predict anchor not found"
src = src.replace(old_pred, new_pred)

open(P, "w").write(src)
print("PATCH OK")
