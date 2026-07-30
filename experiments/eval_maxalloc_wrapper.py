#!/usr/bin/env python
# 包装 eval.py:退出时打印 torch.cuda.max_memory_allocated(分配器真值,替代 nvidia-smi 轮询噪声)
import atexit, os, sys
import torch


@atexit.register
def _report():
    try:
        print(f"MAXALLOC_GB {torch.cuda.max_memory_allocated() / 2**30:.2f}", flush=True)
    except Exception as e:
        print(f"MAXALLOC_FAIL {e}", flush=True)


# mmengine 日志器会周期 reset peak stats,导致 atexit 只读到最后窗口——禁掉 reset 保全局高水位
torch.cuda.reset_max_memory_allocated = lambda *a, **k: None
torch.cuda.reset_peak_memory_stats = lambda *a, **k: None

script = sys.argv[1]
sys.argv = sys.argv[1:]
sys.path.insert(0, os.path.dirname(os.path.abspath(script)) or ".")
code = open(script).read()
g = {"__name__": "__main__", "__file__": script}
exec(compile(code, script, "exec"), g)
