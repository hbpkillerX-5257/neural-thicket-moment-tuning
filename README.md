# Neural Thicket moment-tuning experiment

This repository tests whether a domain specialist can be reached by learning
only the mean and scale of each frozen weight matrix in
`Qwen/Qwen3.5-0.8B`. It uses response-only supervised fine-tuning on GSM8K.

For each selected pretrained matrix `W`, the adapter learns two FP32 scalars:

```text
W' = mu0 + sigma0 * mean_shift + exp(log_scale) * (W - mu0)
```

Both scalars start at zero, so the adapted model is identical to the frozen
model before training. The implementation avoids materializing `W'` and saves
only the adapter scalars.

## Kaggle setup

Create a Kaggle notebook with:

- Accelerator: `GPU T4 x2`
- Internet: enabled

Clone the public repository and install its dependencies:

```bash
git clone https://github.com/hbpkillerX-5257/neural-thicket-moment-tuning.git
cd neural-thicket-moment-tuning
pip install -q -r requirements.txt
```

Restart the notebook kernel if Kaggle had already imported `transformers`
before installation.

## Frozen baseline smoke test

Test ten examples on one GPU before starting the full run:

```bash
python evaluate_gsm8k.py \
  --max-samples 10 \
  --batch-size 2 \
  --output-dir outputs/smoke \
  --overwrite
```

Inspect `outputs/smoke/predictions.jsonl` to confirm that responses and answer
parsing look reasonable.

## Full evaluation on both T4s

```bash
torchrun --standalone --nproc_per_node=2 evaluate_gsm8k.py \
  --batch-size 4 \
  --output-dir outputs/qwen3.5-0.8b-gsm8k \
  --overwrite
```

If this runs out of memory, lower `--batch-size` to `2`. To resume an
interrupted run, execute the exact same command without `--overwrite`:

```bash
torchrun --standalone --nproc_per_node=2 evaluate_gsm8k.py \
  --batch-size 4 \
  --output-dir outputs/qwen3.5-0.8b-gsm8k
```

Do not change the GPU count, batch size, token limits, or output directory when
resuming. The evaluator rejects mismatched run configurations.

## Results

The output directory contains:

- `summary.json`: final accuracy and run configuration
- `predictions.jsonl`: merged per-example outputs
- `predictions.rank*.jsonl`: resumable worker checkpoints

Download the entire output directory from Kaggle. The two files most useful for
analysis are `summary.json` and `predictions.jsonl`.

For unusually long responses, rerun with `--max-new-tokens 512`. Keep the same
generation settings for the frozen model and every adapted model so their
accuracies remain directly comparable.

## Moment-training smoke test

Run a short two-GPU training job before committing to the full experiment:

```bash
torchrun --standalone --nproc_per_node=2 train_gsm8k_moments.py \
  --output-dir outputs/moment-smoke \
  --train-samples 128 \
  --validation-size 32 \
  --epochs 2 \
  --max-steps 20 \
  --gradient-accumulation-steps 2 \
  --save-every-steps 5 \
  --overwrite
```

This updates only the per-matrix moment scalars. Evaluate the smoke adapter on
ten test examples:

```bash
torchrun --standalone --nproc_per_node=2 evaluate_gsm8k.py \
  --max-samples 10 \
  --batch-size 2 \
  --moment-adapter outputs/moment-smoke/adapter-final \
  --output-dir outputs/moment-smoke-eval \
  --overwrite
```

## Full moment training

The defaults use a fixed 500-example validation split, sequence length 512,
per-GPU batch size 2, gradient accumulation 8, effective batch size 32, three
epochs, and learning rate `3e-3`:

```bash
torchrun --standalone --nproc_per_node=2 train_gsm8k_moments.py \
  --output-dir outputs/moment-gsm8k \
  --overwrite
```

The trainer checkpoints every 100 optimizer steps and after every epoch. To
resume, use the same GPU count and arguments, replacing `--overwrite` with
`--resume`:

```bash
torchrun --standalone --nproc_per_node=2 train_gsm8k_moments.py \
  --output-dir outputs/moment-gsm8k \
  --resume
```

The best epoch is selected using validation loss only. Run the official test
set once with that adapter:

```bash
torchrun --standalone --nproc_per_node=2 evaluate_gsm8k.py \
  --batch-size 4 \
  --moment-adapter outputs/moment-gsm8k/adapter-best \
  --output-dir outputs/moment-gsm8k-test \
  --overwrite
```

Training output includes:

- `adapter-best/`: adapter selected by validation loss
- `adapter-final/`: adapter at the final completed step
- `checkpoints/`: resumable adapter and optimizer states
- `data_stats.json`: split sizes and truncation counts
- `run_config.json`: exact experimental configuration

## LoRA comparison (same 2xT4 setup)

`train_gsm8k_lora.py` mirrors the moment trainer: same GSM8K split seed,
validation size, sequence length, batching, epochs, DDP, and checkpointing.
Only the adapter and default learning rate differ (`2e-4`, rank `1`).

Smoke test:

```bash
pip install -q -r requirements.txt

torchrun --standalone --nproc_per_node=2 train_gsm8k_lora.py \
  --output-dir outputs/lora-smoke \
  --train-samples 128 \
  --validation-size 32 \
  --epochs 2 \
  --max-steps 20 \
  --gradient-accumulation-steps 2 \
  --save-every-steps 5 \
  --overwrite
```

Full LoRA training:

```bash
torchrun --standalone --nproc_per_node=2 train_gsm8k_lora.py \
  --output-dir outputs/lora-gsm8k \
  --overwrite
```

If your moment run used a different batch size, pass the same values here so
the effective batch size matches. Resume with `--resume` instead of
`--overwrite`.

Evaluate the validation-selected adapter:

```bash
torchrun --standalone --nproc_per_node=2 evaluate_gsm8k.py \
  --batch-size 4 \
  --lora-adapter outputs/lora-gsm8k/adapter-best \
  --output-dir outputs/lora-gsm8k-test \
  --overwrite
```

## Tests

The local tests verify answer parsing, initialization equivalence, efficient
forward equivalence, frozen-weight gradients, and adapter serialization:

```bash
python -m unittest -v
```
