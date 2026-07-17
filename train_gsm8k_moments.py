#!/usr/bin/env python3
"""Supervised GSM8K training for per-matrix moment adapters.

Two-GPU Kaggle example:
    torchrun --standalone --nproc_per_node=2 train_gsm8k_moments.py \
        --output-dir outputs/moment-gsm8k --overwrite

Resume an interrupted run:
    torchrun --standalone --nproc_per_node=2 train_gsm8k_moments.py \
        --output-dir outputs/moment-gsm8k --resume
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from evaluate_gsm8k import DEFAULT_DATASET, DEFAULT_MODEL, DEFAULT_SYSTEM_PROMPT
from moment_tuning import (
    DEFAULT_TARGET_PREFIX,
    count_moment_parameters,
    inject_moment_adapters,
    load_moment_adapter,
    moment_parameters,
    moment_statistics,
    save_moment_adapter,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--dataset-config", default="main")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/moment-gsm8k"))
    parser.add_argument("--mode", choices=["both", "mean", "scale"], default="both")
    parser.add_argument("--validation-size", type=int, default=500)
    parser.add_argument(
        "--train-samples",
        type=int,
        default=None,
        help="Limit the post-split training set for smoke tests.",
    )
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument(
        "--initial-loss-scale",
        type=float,
        default=128.0,
        help="Initial dynamic loss scale; scalar adapters can overflow at 65536.",
    )
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--save-every-steps",
        type=int,
        default=100,
        help="Write an adapter-only recovery checkpoint every N optimizer steps.",
    )
    parser.add_argument(
        "--no-gradient-checkpointing",
        action="store_true",
        help="Disable activation checkpointing if memory is plentiful.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def distributed_context() -> tuple[int, int, int]:
    return (
        int(os.environ.get("RANK", "0")),
        int(os.environ.get("WORLD_SIZE", "1")),
        int(os.environ.get("LOCAL_RANK", "0")),
    )


def initialize_distributed() -> tuple[Any, Any, int, int, int]:
    import torch
    import torch.distributed as dist

    rank, world_size, local_rank = distributed_context()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for moment training.")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    if world_size > 1:
        try:
            dist.init_process_group(backend="nccl", device_id=device)
        except TypeError:
            dist.init_process_group(backend="nccl")
    return torch, dist, rank, world_size, local_rank


def set_seed(seed: int, rank: int) -> None:
    import torch

    random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)


def run_config(args: argparse.Namespace, world_size: int) -> dict[str, Any]:
    return {
        "model": args.model,
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "mode": args.mode,
        "target_prefix": DEFAULT_TARGET_PREFIX,
        "validation_size": args.validation_size,
        "train_samples": args.train_samples,
        "max_length": args.max_length,
        "batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": (
            args.batch_size * args.gradient_accumulation_steps * world_size
        ),
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "initial_loss_scale": args.initial_loss_scale,
        "max_grad_norm": args.max_grad_norm,
        "seed": args.seed,
        "world_size": world_size,
        "gradient_checkpointing": not args.no_gradient_checkpointing,
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
        "thinking": False,
    }


def prepare_run_directory(
    args: argparse.Namespace,
    config: dict[str, Any],
    rank: int,
    world_size: int,
    dist: Any,
) -> None:
    config_path = args.output_dir / "run_config.json"
    if rank == 0:
        if args.overwrite and args.resume:
            raise ValueError("--overwrite and --resume cannot be used together.")
        if args.overwrite and args.output_dir.exists():
            shutil.rmtree(args.output_dir)
        args.output_dir.mkdir(parents=True, exist_ok=True)

        if config_path.exists():
            previous = json.loads(config_path.read_text(encoding="utf-8"))
            if previous != config:
                raise RuntimeError(
                    "Output directory contains a different training configuration. "
                    "Use another --output-dir or pass --overwrite."
                )
            if not args.resume:
                raise RuntimeError(
                    "Output directory already contains a run. Pass --resume to "
                    "continue it or --overwrite to start over."
                )
        elif args.resume:
            raise RuntimeError("Cannot resume: run_config.json does not exist.")
        else:
            config_path.write_text(
                json.dumps(config, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    if world_size > 1:
        dist.barrier()


def tokenize_example(
    example: dict[str, Any],
    *,
    tokenizer: Any,
    max_length: int,
) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": example["question"]},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    full_text = tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": example["answer"]}],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    if not full_text.startswith(prompt_text):
        raise RuntimeError(
            "Chat template's training prefix differs from generation prefix."
        )
    prompt_ids = tokenizer(
        prompt_text,
        add_special_tokens=False,
    )["input_ids"]
    full_ids = tokenizer(
        full_text,
        add_special_tokens=False,
    )["input_ids"]
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise RuntimeError("Tokenized training prefix differs from generation prefix.")
    if len(prompt_ids) >= max_length:
        raise ValueError(
            f"Prompt uses {len(prompt_ids)} tokens, exceeding max length {max_length}."
        )

    truncated = len(full_ids) > max_length
    input_ids = full_ids[:max_length]
    labels = [-100] * len(prompt_ids) + input_ids[len(prompt_ids) :]
    if not any(label != -100 for label in labels):
        raise RuntimeError("Tokenized example contains no supervised assistant tokens.")
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
        "length": len(input_ids),
        "truncated": truncated,
    }


def prepare_tokenized_data(
    args: argparse.Namespace,
    rank: int,
    world_size: int,
    dist: Any,
) -> None:
    train_path = args.output_dir / "tokenized_train"
    validation_path = args.output_dir / "tokenized_validation"
    stats_path = args.output_dir / "data_stats.json"
    if rank == 0 and not train_path.exists():
        from datasets import load_dataset
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            trust_remote_code=args.trust_remote_code,
        )
        dataset = load_dataset(
            args.dataset,
            args.dataset_config,
            split="train",
        )
        if not 0 < args.validation_size < len(dataset):
            raise ValueError("--validation-size must be between zero and dataset size.")
        split = dataset.train_test_split(
            test_size=args.validation_size,
            seed=args.seed,
            shuffle=True,
        )
        train_data = split["train"]
        validation_data = split["test"]
        if args.train_samples is not None:
            if args.train_samples <= 0:
                raise ValueError("--train-samples must be positive.")
            train_data = train_data.select(
                range(min(args.train_samples, len(train_data)))
            )

        def tokenize(row: dict[str, Any]) -> dict[str, Any]:
            return tokenize_example(
                row,
                tokenizer=tokenizer,
                max_length=args.max_length,
            )

        train_tokenized = train_data.map(
            tokenize,
            remove_columns=train_data.column_names,
            desc="Tokenizing GSM8K training split",
        )
        validation_tokenized = validation_data.map(
            tokenize,
            remove_columns=validation_data.column_names,
            desc="Tokenizing GSM8K validation split",
        )
        stats = {
            "train_examples": len(train_tokenized),
            "validation_examples": len(validation_tokenized),
            "train_truncated": sum(train_tokenized["truncated"]),
            "validation_truncated": sum(validation_tokenized["truncated"]),
            "train_max_tokens": max(train_tokenized["length"]),
            "validation_max_tokens": max(validation_tokenized["length"]),
        }
        train_tokenized = train_tokenized.remove_columns(["length", "truncated"])
        validation_tokenized = validation_tokenized.remove_columns(
            ["length", "truncated"]
        )
        train_tokenized.save_to_disk(train_path)
        validation_tokenized.save_to_disk(validation_path)
        stats_path.write_text(
            json.dumps(stats, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"Prepared data: {json.dumps(stats)}", flush=True)
    if world_size > 1:
        dist.barrier()
    if not train_path.exists() or not validation_path.exists():
        raise RuntimeError("Tokenized dataset cache is incomplete.")


class CausalLMCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        max_length = max(len(example["input_ids"]) for example in examples)
        input_ids = []
        attention_masks = []
        labels = []
        for example in examples:
            padding = max_length - len(example["input_ids"])
            input_ids.append(example["input_ids"] + [self.pad_token_id] * padding)
            attention_masks.append(example["attention_mask"] + [0] * padding)
            labels.append(example["labels"] + [-100] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def load_training_model(
    args: argparse.Namespace, device: Any, resume_dir: Path | None
) -> Any:
    import torch
    from transformers import Qwen3_5ForConditionalGeneration

    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=args.trust_remote_code,
    )
    if resume_dir is None:
        names = inject_moment_adapters(model, mode=args.mode)
    else:
        adapter_config = load_moment_adapter(
            model,
            resume_dir / "adapter",
            expected_base_model=args.model,
        )
        names = adapter_config["module_names"]
        if adapter_config["mode"] != args.mode:
            raise ValueError("Checkpoint adapter mode does not match --mode.")

    model.config.use_cache = False
    if not args.no_gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()
    model.to(device)
    model.train()
    return model, names


def resolve_resume_directory(args: argparse.Namespace) -> Path | None:
    if not args.resume:
        return None
    pointer = args.output_dir / "last_checkpoint.txt"
    if not pointer.exists():
        raise RuntimeError("Cannot resume: last_checkpoint.txt does not exist.")
    checkpoint = args.output_dir / pointer.read_text(encoding="utf-8").strip()
    if not checkpoint.exists():
        raise RuntimeError(f"Cannot resume: checkpoint {checkpoint} is missing.")
    return checkpoint


def save_checkpoint(
    *,
    model: Any,
    optimizer: Any,
    scaler: Any,
    args: argparse.Namespace,
    config: dict[str, Any],
    epoch: int,
    next_batch: int,
    global_step: int,
    best_validation_loss: float | None,
    checkpoint_name: str,
) -> Path:
    import torch

    checkpoint_dir = args.output_dir / "checkpoints" / checkpoint_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_moment_adapter(
        model,
        checkpoint_dir / "adapter",
        base_model=args.model,
        mode=args.mode,
        training_metadata={
            "epoch": epoch,
            "next_batch": next_batch,
            "global_step": global_step,
            "best_validation_loss": best_validation_loss,
            "run_config": config,
        },
    )
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
        },
        checkpoint_dir / "optimizer_state.pt",
    )
    state = {
        "epoch": epoch,
        "next_batch": next_batch,
        "global_step": global_step,
        "best_validation_loss": best_validation_loss,
    }
    (checkpoint_dir / "trainer_state.json").write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    relative = checkpoint_dir.relative_to(args.output_dir)
    (args.output_dir / "last_checkpoint.txt").write_text(
        str(relative) + "\n",
        encoding="utf-8",
    )
    return checkpoint_dir


def load_optimizer_state(
    resume_dir: Path,
    optimizer: Any,
    scaler: Any,
    device: Any,
) -> dict[str, Any]:
    import torch

    state = json.loads((resume_dir / "trainer_state.json").read_text(encoding="utf-8"))
    optimizer_state = torch.load(
        resume_dir / "optimizer_state.pt",
        map_location=device,
        weights_only=False,
    )
    optimizer.load_state_dict(optimizer_state["optimizer"])
    scaler.load_state_dict(optimizer_state["scaler"])
    return state


def validate(
    model: Any, dataloader: Any, device: Any, dist: Any, world_size: int
) -> float:
    import torch

    model.eval()
    totals = torch.zeros(2, dtype=torch.float64, device=device)
    with torch.inference_mode():
        for batch in dataloader:
            batch = {
                key: value.to(device, non_blocking=True) for key, value in batch.items()
            }
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(**batch, use_cache=False)
            token_count = (batch["labels"][:, 1:] != -100).sum()
            totals[0] += outputs.loss.detach().double() * token_count
            totals[1] += token_count
    if world_size > 1:
        dist.all_reduce(totals)
    model.train()
    return (totals[0] / totals[1].clamp_min(1)).item()


def train(args: argparse.Namespace) -> None:
    torch, dist, rank, world_size, local_rank = initialize_distributed()
    device = torch.device(f"cuda:{local_rank}")
    set_seed(args.seed, rank)
    config = run_config(args, world_size)
    prepare_run_directory(args, config, rank, world_size, dist)
    prepare_tokenized_data(args, rank, world_size, dist)

    from datasets import load_from_disk
    from torch.nn.parallel import DistributedDataParallel
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    collator = CausalLMCollator(tokenizer.pad_token_id)
    train_data = load_from_disk(args.output_dir / "tokenized_train")
    validation_data = load_from_disk(args.output_dir / "tokenized_validation")

    # Use DistributedSampler even for a single process so epoch ordering can be
    # reconstructed exactly when skipping already completed batches on resume.
    train_sampler = DistributedSampler(
        train_data,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.seed,
        drop_last=False,
    )
    validation_sampler = DistributedSampler(
        validation_data,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
    )
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=False,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=args.batch_size,
        sampler=validation_sampler,
        shuffle=False,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    resume_dir = resolve_resume_directory(args)
    model, module_names = load_training_model(args, device, resume_dir)
    trainable_parameters = list(moment_parameters(model))
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=0.0,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        init_scale=args.initial_loss_scale,
    )

    start_epoch = 0
    start_batch = 0
    global_step = 0
    best_validation_loss: float | None = None
    if resume_dir is not None:
        state = load_optimizer_state(resume_dir, optimizer, scaler, device)
        start_epoch = int(state["epoch"])
        start_batch = int(state["next_batch"])
        global_step = int(state["global_step"])
        best_validation_loss = state["best_validation_loss"]

    wrapped_model = (
        DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
        if world_size > 1
        else model
    )
    unwrapped_model = wrapped_model.module if world_size > 1 else wrapped_model

    if rank == 0:
        print(
            json.dumps(
                {
                    "matrix_count": len(module_names),
                    "trainable_parameters": count_moment_parameters(unwrapped_model),
                    "total_training_examples": len(train_data),
                    "total_validation_examples": len(validation_data),
                    "steps_per_epoch": math.ceil(
                        len(train_loader) / args.gradient_accumulation_steps
                    ),
                    "resuming_from": str(resume_dir) if resume_dir else None,
                },
                indent=2,
            ),
            flush=True,
        )

    stop_training = args.max_steps is not None and global_step >= args.max_steps
    last_saved_step = -1
    for epoch in range(start_epoch, args.epochs):
        if stop_training:
            break
        train_sampler.set_epoch(epoch)
        optimizer.zero_grad(set_to_none=True)
        epoch_loss_sum = 0.0
        epoch_token_count = 0
        total_batches = len(train_loader)
        epoch_start_batch = start_batch if epoch == start_epoch else 0

        for batch_index, batch in enumerate(train_loader):
            if batch_index < epoch_start_batch:
                continue
            batch = {
                key: value.to(device, non_blocking=True) for key, value in batch.items()
            }
            group_start = (
                batch_index // args.gradient_accumulation_steps
            ) * args.gradient_accumulation_steps
            group_size = min(
                args.gradient_accumulation_steps,
                total_batches - group_start,
            )
            should_step = (
                (batch_index + 1) % args.gradient_accumulation_steps == 0
                or batch_index + 1 == total_batches
            )
            sync_context = (
                nullcontext()
                if should_step or world_size == 1
                else wrapped_model.no_sync()
            )
            with sync_context:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = wrapped_model(**batch, use_cache=False)
                    scaled_loss = outputs.loss / group_size
                scaler.scale(scaled_loss).backward()

            token_count = int((batch["labels"][:, 1:] != -100).sum().item())
            epoch_loss_sum += outputs.loss.detach().float().item() * token_count
            epoch_token_count += token_count

            if should_step:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    trainable_parameters,
                    args.max_grad_norm,
                )
                scale_before_update = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                update_skipped = scaler.get_scale() < scale_before_update
                optimizer.zero_grad(set_to_none=True)
                if not update_skipped:
                    global_step += 1

                if rank == 0 and (
                    update_skipped or global_step <= 5 or global_step % 10 == 0
                ):
                    print(
                        f"epoch={epoch + 1}/{args.epochs} step={global_step} "
                        f"batch={batch_index + 1}/{total_batches} "
                        f"loss={outputs.loss.item():.4f} "
                        f"grad_norm={float(grad_norm):.4f} "
                        f"loss_scale={scaler.get_scale():.1f} "
                        f"update_skipped={update_skipped}",
                        flush=True,
                    )

                if (
                    not update_skipped
                    and args.save_every_steps > 0
                    and global_step % args.save_every_steps == 0
                ):
                    if world_size > 1:
                        dist.barrier()
                    if rank == 0:
                        save_checkpoint(
                            model=unwrapped_model,
                            optimizer=optimizer,
                            scaler=scaler,
                            args=args,
                            config=config,
                            epoch=epoch,
                            next_batch=batch_index + 1,
                            global_step=global_step,
                            best_validation_loss=best_validation_loss,
                            checkpoint_name=f"checkpoint-step-{global_step}",
                        )
                        last_saved_step = global_step
                    if world_size > 1:
                        dist.barrier()

                if (
                    not update_skipped
                    and args.max_steps is not None
                    and global_step >= args.max_steps
                ):
                    stop_training = True
                    break

        if stop_training:
            if world_size > 1:
                dist.barrier()
            if rank == 0 and last_saved_step != global_step:
                save_checkpoint(
                    model=unwrapped_model,
                    optimizer=optimizer,
                    scaler=scaler,
                    args=args,
                    config=config,
                    epoch=epoch,
                    next_batch=batch_index + 1,
                    global_step=global_step,
                    best_validation_loss=best_validation_loss,
                    checkpoint_name=f"checkpoint-step-{global_step}",
                )
            if world_size > 1:
                dist.barrier()
            break

        validation_loss = validate(
            wrapped_model,
            validation_loader,
            device,
            dist,
            world_size,
        )
        loss_totals = torch.tensor(
            [epoch_loss_sum, epoch_token_count],
            dtype=torch.float64,
            device=device,
        )
        if world_size > 1:
            dist.all_reduce(loss_totals)
        training_loss = (loss_totals[0] / loss_totals[1].clamp_min(1)).item()
        improved = (
            best_validation_loss is None or validation_loss < best_validation_loss
        )
        if improved:
            best_validation_loss = validation_loss

        if world_size > 1:
            dist.barrier()
        if rank == 0:
            print(
                json.dumps(
                    {
                        "epoch": epoch + 1,
                        "global_step": global_step,
                        "training_loss": training_loss,
                        "validation_loss": validation_loss,
                        "validation_perplexity": math.exp(min(validation_loss, 20)),
                        "best": improved,
                        "moments": moment_statistics(unwrapped_model),
                    },
                    indent=2,
                ),
                flush=True,
            )
            save_checkpoint(
                model=unwrapped_model,
                optimizer=optimizer,
                scaler=scaler,
                args=args,
                config=config,
                epoch=epoch + 1,
                next_batch=0,
                global_step=global_step,
                best_validation_loss=best_validation_loss,
                checkpoint_name=f"checkpoint-epoch-{epoch + 1}",
            )
            if improved:
                save_moment_adapter(
                    unwrapped_model,
                    args.output_dir / "adapter-best",
                    base_model=args.model,
                    mode=args.mode,
                    training_metadata={
                        "selected_epoch": epoch + 1,
                        "global_step": global_step,
                        "validation_loss": validation_loss,
                        "run_config": config,
                    },
                )
        if world_size > 1:
            dist.barrier()
        start_batch = 0

    if rank == 0:
        save_moment_adapter(
            unwrapped_model,
            args.output_dir / "adapter-final",
            base_model=args.model,
            mode=args.mode,
            training_metadata={
                "global_step": global_step,
                "best_validation_loss": best_validation_loss,
                "stopped_at_max_steps": stop_training,
                "run_config": config,
            },
        )
        print(
            f"Training complete at step {global_step}. "
            f"Final adapter: {args.output_dir / 'adapter-final'}",
            flush=True,
        )
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("--gradient-accumulation-steps must be positive.")
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("--max-steps must be positive.")
    if args.initial_loss_scale <= 0:
        raise ValueError("--initial-loss-scale must be positive.")
    train(args)


if __name__ == "__main__":
    main()
