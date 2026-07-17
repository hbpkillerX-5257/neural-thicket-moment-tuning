# Neural Thicket moment-tuning experiment

The first step is to establish the frozen `Qwen/Qwen3.5-0.8B` baseline on the
official GSM8K test split. `evaluate_gsm8k.py` uses deterministic greedy
decoding, checkpoints every batch, and supports one or two GPUs.

## Kaggle setup

Create a Kaggle notebook with:

- Accelerator: `GPU T4 x2`
- Internet: enabled

Upload this directory as a Kaggle dataset or create the files in the notebook,
then run:

```bash
pip install -q -r requirements.txt
```

Restart the notebook kernel if Kaggle had already imported `transformers`
before installation.

## Smoke test

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
