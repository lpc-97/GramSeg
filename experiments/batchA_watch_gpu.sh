#!/usr/bin/env bash
# Batch A GPU watcher: poll GPU 2 (preferred) then GPU 1 every 5 min.
# Claim rules: >=12GB free AND no ~/.gpu_claim_gpu<N>_* file from another batch.
# On claim: touch ~/.gpu_claim_gpu<N>_batchA, double-check free mem, run driver,
# remove claim when done. Never touches GPU 0/3.
set -u

DRIVER=$HOME/batchA_vitb16.sh
WLOG=$HOME/batchA_watch.log
NEED_MB=12000

log () { echo "$(date +%F_%T) $*" >> "$WLOG"; }

free_mb () {
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" 2>/dev/null | tr -d ' '
}

other_claim () {
    # any claim file for this gpu not owned by batchA
    for f in "$HOME"/.gpu_claim_gpu"$1"_*; do
        [ -e "$f" ] || continue
        case "$f" in *_batchA) ;; *) return 0 ;; esac
    done
    return 1
}

log "watcher start (need ${NEED_MB}MB on GPU 2 or 1)"
while true; do
    for g in 2 1; do
        fm=$(free_mb "$g"); fm=${fm:-0}
        if [ "$fm" -ge "$NEED_MB" ] && ! other_claim "$g"; then
            touch "$HOME/.gpu_claim_gpu${g}_batchA"
            sleep 20
            fm2=$(free_mb "$g"); fm2=${fm2:-0}
            if [ "$fm2" -lt "$NEED_MB" ] || other_claim "$g"; then
                log "gpu$g re-check failed (free=${fm2}MB), releasing"
                rm -f "$HOME/.gpu_claim_gpu${g}_batchA"
                continue
            fi
            log "claimed gpu$g (free=${fm2}MB), launching driver"
            GPU=$g bash "$DRIVER" >> "$WLOG" 2>&1
            rc=$?
            rm -f "$HOME/.gpu_claim_gpu${g}_batchA"
            log "driver exited rc=$rc, claim released"
            exit $rc
        fi
    done
    sleep 300
done
