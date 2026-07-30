#!/bin/bash
# 服务器端离线安装 detectron2(env bl, torch2.6.0+cu124, py3.10)
# 前置:把本地 staging/wheels/ 整目录 scp 到服务器 ~/wheels/
#   scp -r /mnt/e/seg/staging/wheels test@<server>:~/
set -e
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh
conda activate bl

cd ~/wheels
# 先装纯依赖(有 deps/ 子目录则用之;缺的包 --no-index 会明确报错,补下即可)
pip install --no-index --find-links=deps deps/*.whl deps/*.tar.gz 2>/dev/null || true
# 装 detectron2 本体(--no-deps 防止它拉网上的东西)
pip install --no-index --no-deps detectron2-0.6+fd27788pt2.6.0cu124-cp310-cp310-linux_x86_64.whl
# 缺什么依赖会在 import 时暴露:
python -c "import detectron2; from detectron2 import model_zoo; print('detectron2 OK', detectron2.__version__)"
python -c "import torch, detectron2; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
