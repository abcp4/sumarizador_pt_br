# Multi-GPU Training (vast.ai) - T5Gemma Summarization

This folder contains a robust training script for multi-GPU fine-tuning with checkpoint resume support:
- `train_multigpu_vast.py`
- `requirements.txt`
- `train.sh` (example run command)

## 1. Install dependencies

From a `pytorch/pytorch` base image:

```bash
pip install -r /home/spike/Projects/Teses/sumarios_topicos/requirements.txt
```

## 2. Default run (WSD by proportions)

The scheduler is **Warmup -> Stable -> Decay**, controlled by proportions of total estimated update steps.

Example:

```bash
torchrun --nproc_per_node=1 /home/spike/Projects/Teses/sumarios_topicos/train_multigpu_vast.py \
  --data-glob "/home/spike/Projects/Teses/sumarios_topicos/datasets/*/*/*.json" \
  --output-dir "/home/spike/Projects/Teses/sumarios_topicos/checkpoints/t5gemma2-270m-sumarios-ddp" \
  --num-epochs 3 \
  --per-device-train-batch-size 1 \
  --save-every-updates 500 \
  --min-gradient-accumulation-steps 2 \
  --max-gradient-accumulation-steps 16 \
  --grad-accum-growth-factor 2 \
  --warmup-ratio 0.02 \
  --stable-ratio 0.88 \
  --decay-ratio 0.10
```

## 3. Dynamic gradient accumulation

Gradient accumulation is scheduled automatically over training progress:
- starts at `min-gradient-accumulation-steps`
- grows by `grad-accum-growth-factor`
- caps at `max-gradient-accumulation-steps`

Default schedule levels are typically: `2 -> 4 -> 8 -> 16`.

## 4. Checkpoint and resume

The script saves:
- periodic checkpoints: `checkpoints/update_*.pt`
- phase checkpoints: stable start and decay start
- latest pointer: `latest.pt`
- epoch checkpoints: `checkpoints/epoch_*.pt`

Resume automatically:

```bash
torchrun --nproc_per_node=1 /home/spike/Projects/Teses/sumarios_topicos/train_multigpu_vast.py \
  --output-dir "/home/spike/Projects/Teses/sumarios_topicos/checkpoints/t5gemma2-270m-sumarios-ddp" \
  --resume auto
```

Resume from explicit checkpoint:

```bash
torchrun --nproc_per_node=1 /home/spike/Projects/Teses/sumarios_topicos/train_multigpu_vast.py \
  --resume "/home/spike/Projects/Teses/sumarios_topicos/checkpoints/t5gemma2-270m-sumarios-ddp/latest.pt"
```

## 5. Force transition to decay phase after interruption

If training is in stable phase and you want to evaluate a decay strategy immediately:

```bash
torchrun --nproc_per_node=1 /home/spike/Projects/Teses/sumarios_topicos/train_multigpu_vast.py \
  --resume "/home/spike/Projects/Teses/sumarios_topicos/checkpoints/t5gemma2-270m-sumarios-ddp/latest.pt" \
  --force-decay-on-resume \
  --warmup-ratio 0.0 \
  --stable-ratio 0.0 \
  --decay-ratio 1.0 \
  --num-epochs 1
```

Behavior:
- `--force-decay-on-resume` moves scheduler directly to decay mode in the resumed run.
- By default, scheduler state is **not** restored unless you explicitly pass `--resume-load-scheduler-state`.
- Use `--resume-load-scheduler-state` only if you want exact continuation of the old LR schedule.

## 6. Multi-GPU on vast.ai

Increase GPU count with torchrun:

```bash
torchrun --nproc_per_node=4 /home/spike/Projects/Teses/sumarios_topicos/train_multigpu_vast.py [your args]
```

The script uses DDP and handles rank/local_rank/world_size automatically.
