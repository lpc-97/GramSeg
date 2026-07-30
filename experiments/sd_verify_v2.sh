#!/bin/bash
# 看门 v2:以"三大件 blob 全部落盘"为完成判据(不依赖进程存活);
# 下载进程死了且没下完 → 自动重启 hf-mirror 下载;齐了 → sha256 全量重算 → marker → GPU1 CLIPer-L 校准。
exec > ~/sd_verify2.log 2>&1
echo "watcher v2 start $(date)"

CACHE=${CACHE_ROOT:-$HOME/cache}/sd21_cache/models--sd2-community--stable-diffusion-2-1-base
UNET=6dfae3e5f7d459b50f4b0850ead945972c75bb0e1897628933e169eb43974214
TXT=cce6febb0b6d876ee5eb24af35e27e764eb4f9b1d0b7c026c8c3333d4cfc916c
VAE=a1d993488569e928462932c8c38a0760b874d166399b14414135bd9c42df5815

relaunch_dl() {
  echo "relaunching download $(date)"
  nohup env HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 HF_HUB_ETAG_TIMEOUT=60 PYTHONNOUSERSITE=1 \
    ~/miniconda3/envs/bl/bin/huggingface-cli download sd2-community/stable-diffusion-2-1-base \
    --include model_index.json "feature_extractor/*" "scheduler/*" "tokenizer/*" \
    text_encoder/config.json text_encoder/model.safetensors \
    unet/config.json unet/diffusion_pytorch_model.safetensors \
    vae/config.json vae/diffusion_pytorch_model.safetensors \
    --cache-dir ${CACHE_ROOT:-$HOME/cache}/sd21_cache >> ~/sd_download2.log 2>&1 &
}

# 最多等 4h
for i in $(seq 1 240); do
  if [ -f "$CACHE/blobs/$UNET" ] && [ -f "$CACHE/blobs/$TXT" ] && [ -f "$CACHE/blobs/$VAE" ] \
     && ! ls "$CACHE"/blobs/*.incomplete >/dev/null 2>&1; then
    break
  fi
  # 自愈:进程没了但还没齐 → 重启下载
  if ! pgrep -f "huggingface-cli downloa[d]" >/dev/null 2>&1; then
    relaunch_dl; sleep 30
  fi
  sleep 60
done

if ! [ -f "$CACHE/blobs/$UNET" ]; then
  echo "SD21_TIMEOUT: unet still missing $(date)"; exit 1
fi
sleep 5

fail=0
for blob in "$CACHE"/blobs/*; do
  name=$(basename "$blob")
  case "$name" in *.incomplete) echo "INCOMPLETE remains $name"; fail=1; continue;; esac
  if [ ${#name} -eq 64 ]; then
    calc=$(sha256sum "$blob" | awk '{print $1}')
    if [ "$calc" = "$name" ]; then echo "SHA256 OK  $name"; else echo "SHA256 MISMATCH $name got $calc"; fail=1; fi
  else
    echo "skip non-LFS blob $name"
  fi
done
for want in $UNET $TXT $VAE; do
  [ -f "$CACHE/blobs/$want" ] || { echo "MISSING $want"; fail=1; }
done

if [ $fail -ne 0 ]; then echo "SD21_VERIFY_FAILED $(date)"; exit 1; fi
touch ~/sd21_verified.ok
echo "SD21_VERIFIED $(date)"

mkdir -p ~/batchU
cd ~/cliper.code/ovs && PYTHONNOUSERSITE=1 PYTHONPATH=$HOME/cliper.code HF_HUB_OFFLINE=1 \
  CUDA_VISIBLE_DEVICES=1 ~/miniconda3/envs/bl/bin/python main_ovs.py \
  --cfg-path ../scripts/config/vit-l-14/ovs_ade150.yaml \
  --log-path ~/batchU/cliper_calib_log > ~/batchU/cliper_calib.log 2>&1
echo "CLIPER_CALIB_DONE rc=$? $(date)"
