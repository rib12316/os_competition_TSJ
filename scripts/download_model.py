"""下载 HuggingFace 模型到 ``models/``（默认走 hf-mirror 国内镜像）。

用法::

    python scripts/download_model.py Qwen/Qwen2.5-7B-Instruct models/Qwen2.5-7B-Instruct

环境变量 ``HF_ENDPOINT`` 未设时默认 ``https://hf-mirror.com``。可后台挂起运行：

    nohup python scripts/download_model.py Qwen/Qwen2.5-7B-Instruct models/Qwen2.5-7B-Instruct \
        > logs/_installs/download.log 2>&1 &
"""

from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from huggingface_hub import snapshot_download  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="下载 HF 模型到 models/")
    p.add_argument("repo_id", help="如 Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("local_dir", help="本地目标目录，如 models/Qwen2.5-7B-Instruct")
    p.add_argument("--max-workers", type=int, default=4, help="并发下载数")
    args = p.parse_args()

    endpoint = os.environ["HF_ENDPOINT"]
    print(
        f"[download] {args.repo_id} -> {args.local_dir} (endpoint={endpoint}, workers={args.max_workers})",
        flush=True,
    )
    t0 = time.time()
    path = snapshot_download(
        repo_id=args.repo_id,
        local_dir=args.local_dir,
        max_workers=args.max_workers,
    )
    print(f"[download] DONE -> {path} ({round(time.time() - t0)}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
