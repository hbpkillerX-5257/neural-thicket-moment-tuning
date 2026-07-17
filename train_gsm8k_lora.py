#!/usr/bin/env python3
"""Supervised GSM8K training with LoRA, matched to train_gsm8k_moments.py.

Two-GPU Kaggle example:
    torchrun --standalone --nproc_per_node=2 train_gsm8k_lora.py \
        --output-dir outputs/lora-gsm8k --overwrite

Resume an interrupted run:
    torchrun --standalone --nproc_per_node=2 train_gsm8k_lora.py \
        --output-dir outputs/lora-gsm8k --resume

Hyperparameters intentionally mirror the moment trainer (data split, seed,
sequence length, batching, epochs). LoRA uses a lower default learning rate.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from torch import nn

from evaluate_gsm8k import DEFAULT_DATASET, DEFAULT_MODEL
from moment_tuning import DEFAULT_TARGET_PREFIX
from train_gsm8k_moments import (
    CausalLMCollator,
    initialize_distributed,
    load_optimizer_state,
    prepare_run_directory,
    prepare_tokenized_data,
    resolve_resume_directory,
    set_seed,
    validate,
)

# Same linear suffixes as the moment adapter injection under language layers.
LORA_TARGET_SUFFIXES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_b",
    "in_proj_a",
    "out_proj",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--dataset-config", default="main")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/lora-gsm8k"))
    parser.add_argument("--lora-r", type=int, default=1)
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=None,
        help="LoRA alpha. Defaults to 2 * --lora-r.",
    )
    parser.add_argument("--lora-dropout", type=float, default=0.0)
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
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="LoRA learning rate (lower than moment-tuning's 3e-3).",
    )
    parser.add_argument(
        "--initial-loss-scale",
        type=float,
        default=65536.0,
        help="Initial GradScaler loss scale. LoRA can use the PyTorch default.",
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


def run_config(args: argparse.Namespace, world_size: int) -> dict[str, Any]:
    lora_alpha = args.lora_alpha if args.lora_alpha is not None else 2 * args.lora_r
    return {
        "method": "lora",
        "model": args.model,
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "target_prefix": DEFAULT_TARGET_PREFIX,
        "lora_r": args.lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lora_target_suffixes": list(LORA_TARGET_SUFFIXES),
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
        "thinking": False,
    }


def discover_lora_targets(model: nn.Module) -> list[str]:
    """Match moment-tuning: all language-layer Linears under DEFAULT_TARGET_PREFIX."""
    targets: list[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not name.startswith(DEFAULT_TARGET_PREFIX):
            continue
        if not any(name.endswith(suffix) for suffix in LORA_TARGET_SUFFIXES):
            continue
        targets.append(name)
    if not targets:
        raise ValueError(
            f"No LoRA targets found under {DEFAULT_TARGET_PREFIX!r} with "
            f"suffixes {LORA_TARGET_SUFFIXES}."
        )
    return targets


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


def save_lora_adapter(
    model: Any,
    output_dir: str | Path,
    *,
    training_metadata: dict[str, Any] | None = None,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path)
    metadata = {
        "format": "peft-lora",
        "trainable_parameter_count": count_trainable_parameters(model),
        "training": training_metadata or {},
    }
    (output_path / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def load_training_model(
    args: argparse.Namespace,
    device: Any,
    resume_dir: Path | None,
) -> tuple[Any, list[str]]:
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import Qwen3_5ForConditionalGeneration

    from peft_compat import disable_incompatible_torchao

    if disable_incompatible_torchao():
        print(
            "Disabled incompatible torchao for PEFT LoRA; using nn.Linear adapters.",
            flush=True,
        )

    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=args.trust_remote_code,
    )
    targets = discover_lora_targets(model)
    lora_alpha = args.lora_alpha if args.lora_alpha is not None else 2 * args.lora_r

    if resume_dir is None:
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=targets,
        )
        model = get_peft_model(model, peft_config)
    else:
        model = PeftModel.from_pretrained(
            model,
            resume_dir / "adapter",
            is_trainable=True,
        )

    model.config.use_cache = False
    if not args.no_gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()
    model.to(device)
    model.train()
    return model, targets


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
    save_lora_adapter(
        model,
        checkpoint_dir / "adapter",
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
    model, target_names = load_training_model(args, device, resume_dir)
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
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
                    "method": "lora",
                    "matrix_count": len(target_names),
                    "trainable_parameters": count_trainable_parameters(unwrapped_model),
                    "lora_r": args.lora_r,
                    "lora_alpha": (
                        args.lora_alpha
                        if args.lora_alpha is not None
                        else 2 * args.lora_r
                    ),
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
                        "trainable_parameters": count_trainable_parameters(
                            unwrapped_model
                        ),
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
                if (args.output_dir / "adapter-best").exists():
                    shutil.rmtree(args.output_dir / "adapter-best")
                save_lora_adapter(
                    unwrapped_model,
                    args.output_dir / "adapter-best",
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
        if (args.output_dir / "adapter-final").exists():
            shutil.rmtree(args.output_dir / "adapter-final")
        save_lora_adapter(
            unwrapped_model,
            args.output_dir / "adapter-final",
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
    if args.lora_r <= 0:
        raise ValueError("--lora-r must be positive.")
    if args.lora_alpha is not None and args.lora_alpha <= 0:
        raise ValueError("--lora-alpha must be positive.")
    train(args)


if __name__ == "__main__":
    main()
