#!/usr/bin/env python3
"""Evaluate a Hugging Face chat model on the official GSM8K test split.

Single GPU:
    python evaluate_gsm8k.py --max-samples 100 --overwrite

Two GPUs:
    torchrun --standalone --nproc_per_node=2 evaluate_gsm8k.py --overwrite

Each distributed worker loads one model replica and evaluates a disjoint shard.
Results are checkpointed after every batch, so an interrupted run can resume by
rerunning the same command without --overwrite.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "Qwen/Qwen3.5-0.8B"
DEFAULT_DATASET = "openai/gsm8k"
DEFAULT_SYSTEM_PROMPT = (
    "You are a careful grade-school mathematics solver. Show your reasoning, "
    "then finish with exactly one line in the form: #### <numeric answer>"
)

NUMBER_PATTERN = re.compile(
    r"(?<![\w.])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:/\d+(?:\.\d+)?)?"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--dataset-config", default="main")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/qwen3.5-0.8b-gsm8k"),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Evaluate only the first N examples; default evaluates the full split.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-input-tokens", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument(
        "--moment-adapter",
        type=Path,
        default=None,
        help="Optional directory containing moment adapter_config.json and weights.",
    )
    parser.add_argument(
        "--lora-adapter",
        type=Path,
        default=None,
        help="Optional PEFT LoRA adapter directory (adapter_config.json + weights).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing predictions in output-dir instead of resuming.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow custom Hugging Face model code if the model requires it.",
    )
    return parser.parse_args()


def distributed_context() -> tuple[int, int, int]:
    """Return (rank, world_size, local_rank) from torchrun environment."""
    return (
        int(os.environ.get("RANK", "0")),
        int(os.environ.get("WORLD_SIZE", "1")),
        int(os.environ.get("LOCAL_RANK", "0")),
    )


def canonical_number(value: str | None) -> Fraction | None:
    """Convert common GSM8K numeric forms to an exact comparable value."""
    if value is None:
        return None

    cleaned = (
        value.strip().replace(",", "").replace("$", "").replace("%", "").rstrip(".")
    )
    try:
        if "/" in cleaned:
            numerator, denominator = cleaned.split("/", maxsplit=1)
            return Fraction(Decimal(numerator)) / Fraction(Decimal(denominator))
        return Fraction(Decimal(cleaned))
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return None


def extract_gold(answer: str) -> tuple[str | None, Fraction | None]:
    """Extract the number following GSM8K's #### answer marker."""
    match = re.search(r"####\s*([^\n]+)", answer)
    if not match:
        return None, None
    candidates = NUMBER_PATTERN.findall(match.group(1))
    raw = candidates[-1] if candidates else None
    return raw, canonical_number(raw)


def extract_prediction(response: str) -> tuple[str | None, Fraction | None, str]:
    """Extract an answer with marker-first and last-number fallback parsing."""
    marker_matches = re.findall(r"####\s*([^\n]+)", response)
    if marker_matches:
        candidates = NUMBER_PATTERN.findall(marker_matches[-1])
        if candidates:
            raw = candidates[-1]
            return raw, canonical_number(raw), "####"

    boxed_matches = re.findall(r"\\boxed\{([^{}]+)\}", response)
    if boxed_matches:
        candidates = NUMBER_PATTERN.findall(boxed_matches[-1])
        if candidates:
            raw = candidates[-1]
            return raw, canonical_number(raw), "boxed"

    final_matches = re.findall(
        r"(?:final\s+answer|answer\s+is)\s*[:=]?\s*([^\n]+)",
        response,
        flags=re.IGNORECASE,
    )
    if final_matches:
        candidates = NUMBER_PATTERN.findall(final_matches[-1])
        if candidates:
            raw = candidates[-1]
            return raw, canonical_number(raw), "final-answer"

    candidates = NUMBER_PATTERN.findall(response)
    if candidates:
        raw = candidates[-1]
        return raw, canonical_number(raw), "last-number"
    return None, None, "unparsed"


def build_prompt(tokenizer: Any, question: str) -> str:
    messages = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def load_existing(path: Path) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                records[int(record["index"])] = record
    return records


def append_records(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _hash_paths(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def moment_adapter_descriptor(adapter_dir: Path | None) -> dict[str, Any] | None:
    if adapter_dir is None:
        return None
    from moment_tuning import ADAPTER_CONFIG_NAME, ADAPTER_WEIGHTS_NAME

    config_path = adapter_dir / ADAPTER_CONFIG_NAME
    weights_path = adapter_dir / ADAPTER_WEIGHTS_NAME
    if not config_path.exists() or not weights_path.exists():
        raise FileNotFoundError(
            f"{adapter_dir} must contain {ADAPTER_CONFIG_NAME} and "
            f"{ADAPTER_WEIGHTS_NAME}."
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return {
        "path": str(adapter_dir),
        "sha256": _hash_paths([config_path, weights_path]),
        "base_model": config.get("base_model"),
        "mode": config.get("mode"),
        "module_count": config.get("module_count"),
        "trainable_parameter_count": config.get("trainable_parameter_count"),
    }


def lora_adapter_descriptor(adapter_dir: Path | None) -> dict[str, Any] | None:
    if adapter_dir is None:
        return None
    config_path = adapter_dir / "adapter_config.json"
    weight_candidates = [
        adapter_dir / "adapter_model.safetensors",
        adapter_dir / "adapter_model.bin",
    ]
    weights_path = next((path for path in weight_candidates if path.exists()), None)
    if not config_path.exists() or weights_path is None:
        raise FileNotFoundError(
            f"{adapter_dir} must contain adapter_config.json and "
            "adapter_model.safetensors (or .bin)."
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    metadata_path = adapter_dir / "training_metadata.json"
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.exists()
        else {}
    )
    return {
        "path": str(adapter_dir),
        "sha256": _hash_paths([config_path, weights_path]),
        "peft_type": config.get("peft_type"),
        "r": config.get("r"),
        "lora_alpha": config.get("lora_alpha"),
        "trainable_parameter_count": metadata.get("trainable_parameter_count"),
    }


def make_run_config(args: argparse.Namespace, world_size: int) -> dict[str, Any]:
    if args.moment_adapter is not None and args.lora_adapter is not None:
        raise ValueError("Pass only one of --moment-adapter or --lora-adapter.")
    return {
        "model": args.model,
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "max_samples": args.max_samples,
        "batch_size_per_gpu": args.batch_size,
        "max_input_tokens": args.max_input_tokens,
        "max_new_tokens": args.max_new_tokens,
        "world_size": world_size,
        "decoding": "greedy",
        "thinking": False,
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
        "moment_adapter": moment_adapter_descriptor(args.moment_adapter),
        "lora_adapter": lora_adapter_descriptor(args.lora_adapter),
    }


def prepare_output(
    args: argparse.Namespace,
    config: dict[str, Any],
    rank: int,
    world_size: int,
    dist: Any,
) -> Path:
    output_dir = args.output_dir
    config_path = output_dir / "run_config.json"

    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        if args.overwrite:
            for path in output_dir.glob("predictions.rank*.jsonl"):
                path.unlink()
            for name in ("predictions.jsonl", "summary.json"):
                path = output_dir / name
                if path.exists():
                    path.unlink()

        if config_path.exists() and not args.overwrite:
            previous = json.loads(config_path.read_text(encoding="utf-8"))
            if previous != config:
                raise RuntimeError(
                    "Existing output has a different run configuration. "
                    "Use another --output-dir or pass --overwrite."
                )
        else:
            config_path.write_text(
                json.dumps(config, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

    if world_size > 1:
        dist.barrier()

    # All ranks validate after rank zero creates the file. This also surfaces a
    # clear error if a resumed run uses a different torchrun world size.
    existing_config = json.loads(config_path.read_text(encoding="utf-8"))
    if existing_config != config:
        raise RuntimeError("Run configuration does not match run_config.json.")
    return output_dir / f"predictions.rank{rank}.jsonl"


def load_qwen_model(
    model_id: str,
    device: Any,
    trust_remote_code: bool,
    moment_adapter: Path | None = None,
    lora_adapter: Path | None = None,
) -> tuple[Any, Any]:
    """Load Qwen3.5's multimodal wrapper for text-only generation."""
    import torch
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    if moment_adapter is not None and lora_adapter is not None:
        raise ValueError("Pass only one of moment_adapter or lora_adapter.")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
    )
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=trust_remote_code,
    )
    if moment_adapter is not None:
        from moment_tuning import load_moment_adapter, moment_statistics

        load_moment_adapter(
            model,
            moment_adapter,
            expected_base_model=model_id,
        )
        print(
            f"Loaded moment adapter {moment_adapter}: "
            f"{json.dumps(moment_statistics(model))}",
            flush=True,
        )
    if lora_adapter is not None:
        from peft_compat import disable_incompatible_torchao

        disable_incompatible_torchao()
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, lora_adapter)
        trainable = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        print(
            f"Loaded LoRA adapter {lora_adapter} "
            f"(trainable_when_loaded={trainable})",
            flush=True,
        )
    model.to(device)
    model.eval()

    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


def evaluate_worker(args: argparse.Namespace) -> None:
    import torch
    import torch.distributed as dist
    from datasets import load_dataset

    rank, world_size, local_rank = distributed_context()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this evaluator.")

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    device = torch.device(f"cuda:{local_rank}")

    config = make_run_config(args, world_size)
    rank_path = prepare_output(args, config, rank, world_size, dist)
    completed = load_existing(rank_path)

    dataset = load_dataset(
        args.dataset,
        args.dataset_config,
        split=args.split,
    )
    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError("--max-samples must be positive.")
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    assigned_indices = [
        index
        for index in range(len(dataset))
        if index % world_size == rank and index not in completed
    ]

    if rank == 0:
        print(
            f"Evaluating {len(dataset)} examples with {world_size} GPU(s); "
            f"batch size per GPU={args.batch_size}.",
            flush=True,
        )
    print(
        f"[rank {rank}] {len(assigned_indices)} examples remain; "
        f"{len(completed)} already checkpointed.",
        flush=True,
    )

    if assigned_indices:
        model, tokenizer = load_qwen_model(
            args.model,
            device,
            args.trust_remote_code,
            moment_adapter=args.moment_adapter,
            lora_adapter=args.lora_adapter,
        )
        started = time.monotonic()

        for offset in range(0, len(assigned_indices), args.batch_size):
            batch_indices = assigned_indices[offset : offset + args.batch_size]
            rows = [dataset[index] for index in batch_indices]
            prompts = [build_prompt(tokenizer, row["question"]) for row in rows]

            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_input_tokens,
            ).to(device)

            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            prompt_length = inputs["input_ids"].shape[1]
            generated_only = generated[:, prompt_length:]
            responses = tokenizer.batch_decode(
                generated_only,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            output_records = []
            for index, row, response in zip(batch_indices, rows, responses):
                gold_raw, gold_value = extract_gold(row["answer"])
                predicted_raw, predicted_value, parser = extract_prediction(response)
                output_records.append(
                    {
                        "index": index,
                        "question": row["question"],
                        "gold_answer": row["answer"],
                        "gold_numeric": gold_raw,
                        "response": response,
                        "predicted_numeric": predicted_raw,
                        "parser": parser,
                        "correct": (
                            gold_value is not None
                            and predicted_value is not None
                            and gold_value == predicted_value
                        ),
                    }
                )

            append_records(rank_path, output_records)
            processed = min(offset + len(batch_indices), len(assigned_indices))
            rank_correct = sum(
                int(record["correct"])
                for record in list(completed.values()) + output_records
            )
            elapsed = time.monotonic() - started
            print(
                f"[rank {rank}] {processed}/{len(assigned_indices)} new examples "
                f"({elapsed:.0f}s elapsed, latest checkpoint saved, "
                f"latest batch correct={sum(r['correct'] for r in output_records)}/"
                f"{len(output_records)}, prior correct={rank_correct - sum(r['correct'] for r in output_records)})",
                flush=True,
            )

            for record in output_records:
                completed[int(record["index"])] = record

    if world_size > 1:
        dist.barrier()

    if rank == 0:
        merge_results(args.output_dir, len(dataset), world_size, config)

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


def merge_results(
    output_dir: Path,
    expected_count: int,
    world_size: int,
    config: dict[str, Any],
) -> None:
    all_records: dict[int, dict[str, Any]] = {}
    for rank in range(world_size):
        all_records.update(load_existing(output_dir / f"predictions.rank{rank}.jsonl"))

    ordered = [all_records[index] for index in sorted(all_records)]
    merged_path = output_dir / "predictions.jsonl"
    with merged_path.open("w", encoding="utf-8") as handle:
        for record in ordered:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    correct = sum(int(record["correct"]) for record in ordered)
    unparsed = sum(record["parser"] == "unparsed" for record in ordered)
    fallback = sum(record["parser"] != "####" for record in ordered)
    summary = {
        **config,
        "evaluated": len(ordered),
        "expected": expected_count,
        "complete": len(ordered) == expected_count,
        "correct": correct,
        "accuracy": correct / len(ordered) if ordered else 0.0,
        "unparsed": unparsed,
        "non_marker_parses": fallback,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2), flush=True)
    if len(ordered) != expected_count:
        raise RuntimeError(
            f"Merged {len(ordered)} predictions but expected {expected_count}."
        )


def main() -> None:
    args = parse_args()
    evaluate_worker(args)


if __name__ == "__main__":
    main()
