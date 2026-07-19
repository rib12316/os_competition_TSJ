# models — 模型权重

本目录存放本地模型权重。**权重文件不入 git**（见根 `.gitignore` 的 `models/*`），仅本说明入库。
来源：**ModelScope**（本环境 HuggingFace 不可达，实测 ModelScope 可达且快）。用 `uv pip install modelscope` 后通过 `snapshot_download` 下载。

## 下载约定

```bash
cd /data/os_competition_TSJ && source .venv/bin/activate
python -c "from modelscope import snapshot_download as s; print(s('<org>/<model>', local_dir='models/<name>'))"
```

下载日志：`logs/_installs/<ts>_model-download-<name>/download.log`。

## 当前 / 预期模型

| 目录 | ModelScope 仓库 | 用途 | 状态 |
|---|---|---|---|
| `Qwen3-0.6B/` | `Qwen/Qwen3-0.6B` | 冒烟/最小验证（~1.2GB） | ✅ 已下 |
| `Qwen2.5-7B-Instruct/` | `Qwen/Qwen2.5-7B-Instruct` | MVP 主模型 | ⏳ 待下 |
| `MiniCPM3-4B/` | `OpenBMB/MiniCPM3-4B` | 多模型评分项 | ⏳ 待下 |

> 注：vLLM 加载模型时用绝对/相对路径指向 `models/<name>`，例如 `vllm serve models/Qwen3-0.6B ...`。
