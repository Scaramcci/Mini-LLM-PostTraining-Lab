# Experiment Log

每次实验都记录在这里。失败实验也要记录，因为失败原因本身就是学习成果。


## exp-001: nanoGPT tiny Shakespeare smoke test

### Date

2026-05-25

### Device

MacBook Pro 14-inch, Apple M4 Pro, MPS backend

### Command

```bash
cd 01_nanogpt_pretrain
./scripts/run_train_smoke.sh mps 1000
```

### Config

- dataset: `shakespeare_char`
- out_dir: `out-shakespeare-char`
- max_iters: `1000`
- device: `mps`
- compile: `False`
- batch_size: `64`
- block_size: `256`
- gradient_accumulation_steps: `1`
- tokens_per_iter: `16,384`
- model: 6 layers, 6 heads, 384 embedding dim
- parameters: 10.65M
- learning_rate: `1e-3`
- lr_decay_iters: `5000`
- min_lr: `1e-4`
- warmup_iters: `100`
- dropout: `0.2`
- eval_interval: `250`
- eval_iters: `200`
- log_interval: `10`

### Result

- step 0: train loss `4.2874`, val loss `4.2823`
- step 250: train loss `1.9639`, val loss `2.0676`
- step 500: train loss `1.5303`, val loss `1.7338`
- step 750: train loss `1.3658`, val loss `1.5935`
- step 1000: train loss `1.2720`, val loss `1.5204`
- final logged batch loss at iter 1000: `1.3378`
- checkpoint saved: `01_nanogpt_pretrain/nanoGPT/out-shakespeare-char/ckpt.pt`
- loss CSV: `01_nanogpt_pretrain/nanoGPT/out-shakespeare-char/loss_history.csv`
- loss curve:

![exp-001 loss curve](../01_nanogpt_pretrain/nanoGPT/out-shakespeare-char/loss_curve.png)

### Problems

- `torch.cuda.amp.GradScaler` produced a warning on MPS because CUDA is not available. PyTorch disabled the CUDA scaler automatically, so training continued normally.
- Evaluation steps were much slower than normal training iterations because each evaluation used 200 train batches and 200 val batches. Regular iterations were about 440-490 ms, while evaluation intervals took about 51-56 s.
- MFU stayed low at about 0.70%-0.85%, which is expected for this small model and MPS smoke-test setup.

### Notes

- The first training run completed successfully on MPS for 1000 iterations.
- Loss decreased clearly throughout training: validation loss went from `4.2823` to `1.5204`, showing the model learned the character-level Shakespeare distribution.
- The gap between final train loss (`1.2720`) and val loss (`1.5204`) suggests mild overfitting has started, which matches the tiny Shakespeare setup and the config comment.
- The generated loss curve confirms a steady downward trend with small batch-level fluctuations.
