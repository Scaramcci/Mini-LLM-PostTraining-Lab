# Shakespeare Char Smoke Test Config

这是 Stage 01 第一轮最小测试，不追求效果，只确认流程。

## Suggested Command

```bash
cd 01_nanogpt_pretrain/nanoGPT
python train.py config/train_shakespeare_char.py --device=cpu --compile=False --max_iters=200
```

## If Using MPS

```bash
cd 01_nanogpt_pretrain/nanoGPT
python train.py config/train_shakespeare_char.py --device=mps --compile=False --max_iters=1000
```

## If Using 4090

```bash
cd 01_nanogpt_pretrain/nanoGPT
python train.py config/train_shakespeare_char.py --device=cuda --compile=False --max_iters=2000
```

## Record

- device:
- max_iters:
- final train loss:
- final val loss:
- sample:
