# Environment Setup

记录本地和 GPU 服务器的环境配置。每次环境有变化，都写在这里。

## Local Machine

```text
OS:
Python:
Package manager:
CUDA:
GPU:
```

## First Local Setup

建议先创建虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

检查环境：

```bash
python scripts/check_env.py
```

当前机器备注：

```text
本机 shell 里 `python` 暂不可用，但 `python3` 可用。
创建并激活 `.venv` 后，通常可以在虚拟环境里使用 `python`。
```

## GPU Server Template

```text
Provider:
GPU:
CUDA:
Python:
PyTorch:
Disk:
Hourly price:
```

## Notes

- Stage 01 可以先在本地跑 tiny Shakespeare。
- 到 SFT / DPO / GRPO 时，再按小时租 4090 或更高显卡。
- 云服务器上下载的模型和数据通常在 `~/.cache/huggingface/`，不需要提交到 GitHub。
