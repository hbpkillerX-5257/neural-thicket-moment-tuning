"""Per-weight-matrix mean and scale adapters for frozen linear layers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn.functional as F
from torch import nn

ADAPTER_CONFIG_NAME = "adapter_config.json"
ADAPTER_WEIGHTS_NAME = "adapter_model.safetensors"
DEFAULT_TARGET_PREFIX = "model.language_model.layers."
FORMAT_VERSION = 1
VALID_MODES = {"both", "mean", "scale"}


class _MomentAffine(torch.autograd.Function):
    """Apply scalar affine coefficients with FP32 scalar-gradient reductions."""

    @staticmethod
    def forward(
        ctx: Any,
        base: torch.Tensor,
        input_sum: torch.Tensor,
        scale: torch.Tensor,
        shift: torch.Tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(base, input_sum, scale, shift)
        return scale.to(dtype=base.dtype) * base + shift.to(
            dtype=base.dtype
        ) * input_sum.to(dtype=base.dtype)

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        base, input_sum, scale, shift = ctx.saved_tensors
        grad_output_fp32 = grad_output.float()

        grad_base = grad_output * scale.to(dtype=grad_output.dtype)
        grad_input_sum = (grad_output_fp32.sum(dim=-1, keepdim=True) * shift).to(
            dtype=input_sum.dtype
        )
        grad_scale = (grad_output_fp32 * base.float()).sum().reshape_as(scale)
        grad_shift = (grad_output_fp32 * input_sum.float()).sum().reshape_as(shift)
        return grad_base, grad_input_sum, grad_scale, grad_shift


class MomentLinear(nn.Module):
    """A frozen linear layer with trainable global weight mean and scale.

    Given the frozen pretrained matrix ``W`` with moments ``mu`` and ``sigma``,
    this module implements

        W' = mu + sigma * mean_shift + exp(log_scale) * (W - mu).

    It evaluates the equivalent expression without constructing ``W'``:

        x W'^T = a * (x W^T) + b * sum(x)

    where ``a = exp(log_scale)`` and
    ``b = mu * (1 - a) + sigma * mean_shift``.
    """

    def __init__(self, linear: nn.Linear, mode: str = "both") -> None:
        super().__init__()
        if not isinstance(linear, nn.Linear):
            raise TypeError(f"Expected nn.Linear, got {type(linear).__name__}.")
        if mode not in VALID_MODES:
            raise ValueError(
                f"mode must be one of {sorted(VALID_MODES)}, got {mode!r}."
            )

        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.mode = mode

        # Reuse the exact pretrained parameters instead of making another copy.
        self.weight = linear.weight
        self.weight.requires_grad_(False)
        if linear.bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = linear.bias
            self.bias.requires_grad_(False)

        weight_fp32 = self.weight.detach().float()
        self.register_buffer("pretrained_mean", weight_fp32.mean())
        self.register_buffer("pretrained_std", weight_fp32.std(unbiased=False))

        self.mean_shift = nn.Parameter(
            torch.zeros((), dtype=torch.float32),
            requires_grad=mode in {"both", "mean"},
        )
        self.log_scale = nn.Parameter(
            torch.zeros((), dtype=torch.float32),
            requires_grad=mode in {"both", "scale"},
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, mode={self.mode!r}"
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base = F.linear(inputs, self.weight, bias=None)

        # Keep the learned scalars in FP32 for optimization, then cast only the
        # two coefficients so full activations retain the model's working dtype.
        scale_fp32 = self.log_scale.exp()
        shift_fp32 = (
            self.pretrained_mean * (1.0 - scale_fp32)
            + self.pretrained_std * self.mean_shift
        )
        # The reduction is tiny compared with a second matrix multiplication.
        # Accumulating it in FP32 avoids overflow and cancellation in FP16.
        input_sum = inputs.float().sum(dim=-1, keepdim=True)
        if torch.is_grad_enabled() and (
            scale_fp32.requires_grad or shift_fp32.requires_grad
        ):
            # A regular FP16 scalar multiplication reduces its scalar gradient
            # in FP16 and overflows on large matrices. This custom backward
            # preserves FP16 activations but performs those reductions in FP32.
            output = _MomentAffine.apply(
                base,
                input_sum,
                scale_fp32,
                shift_fp32,
            )
        else:
            output = scale_fp32.to(dtype=base.dtype) * base + shift_fp32.to(
                dtype=base.dtype
            ) * input_sum.to(dtype=base.dtype)
        if self.bias is not None:
            output = output + self.bias
        return output

    @torch.no_grad()
    def materialized_weight(self) -> torch.Tensor:
        """Return the adapted weight, primarily for verification or merging."""
        scale = self.log_scale.exp().to(
            device=self.weight.device,
            dtype=self.weight.dtype,
        )
        mean = self.pretrained_mean.to(
            device=self.weight.device,
            dtype=self.weight.dtype,
        )
        std = self.pretrained_std.to(
            device=self.weight.device,
            dtype=self.weight.dtype,
        )
        mean_shift = self.mean_shift.to(
            device=self.weight.device,
            dtype=self.weight.dtype,
        )
        return mean + std * mean_shift + scale * (self.weight - mean)


def _set_submodule(model: nn.Module, name: str, module: nn.Module) -> None:
    parent_name, _, child_name = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, child_name, module)


def inject_moment_adapters(
    model: nn.Module,
    *,
    mode: str = "both",
    target_prefix: str = DEFAULT_TARGET_PREFIX,
    module_names: list[str] | None = None,
) -> list[str]:
    """Freeze a model and replace selected language-layer linears in-place."""
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {sorted(VALID_MODES)}, got {mode!r}.")

    model.requires_grad_(False)
    allowed = set(module_names) if module_names is not None else None
    targets = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
        and not isinstance(module, MomentLinear)
        and name.startswith(target_prefix)
        and (allowed is None or name in allowed)
    ]

    if allowed is not None:
        found = {name for name, _ in targets}
        missing = allowed - found
        if missing:
            raise ValueError(
                "Adapter references linear modules absent from the base model: "
                + ", ".join(sorted(missing))
            )
    if not targets:
        raise ValueError(
            f"No nn.Linear modules found under target prefix {target_prefix!r}."
        )

    for name, linear in targets:
        _set_submodule(model, name, MomentLinear(linear, mode=mode))
    return [name for name, _ in targets]


def named_moment_modules(model: nn.Module) -> Iterator[tuple[str, MomentLinear]]:
    for name, module in model.named_modules():
        if isinstance(module, MomentLinear):
            yield name, module


def moment_parameters(model: nn.Module) -> Iterator[nn.Parameter]:
    for _, module in named_moment_modules(model):
        if module.mean_shift.requires_grad:
            yield module.mean_shift
        if module.log_scale.requires_grad:
            yield module.log_scale


def count_moment_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in moment_parameters(model))


def moment_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for name, module in named_moment_modules(model):
        state[f"{name}.mean_shift"] = module.mean_shift.detach().float().cpu()
        state[f"{name}.log_scale"] = module.log_scale.detach().float().cpu()
    if not state:
        raise ValueError("The model does not contain any MomentLinear modules.")
    return state


def moment_statistics(model: nn.Module) -> dict[str, float | int]:
    modules = list(named_moment_modules(model))
    if not modules:
        return {
            "module_count": 0,
            "trainable_parameter_count": 0,
        }

    means = torch.stack(
        [module.mean_shift.detach().float().cpu() for _, module in modules]
    )
    scales = torch.stack(
        [module.log_scale.detach().float().cpu().exp() for _, module in modules]
    )
    return {
        "module_count": len(modules),
        "trainable_parameter_count": count_moment_parameters(model),
        "mean_shift_min": means.min().item(),
        "mean_shift_max": means.max().item(),
        "mean_shift_abs_mean": means.abs().mean().item(),
        "scale_min": scales.min().item(),
        "scale_max": scales.max().item(),
        "scale_abs_delta_mean": (scales - 1.0).abs().mean().item(),
    }


def save_moment_adapter(
    model: nn.Module,
    output_dir: str | Path,
    *,
    base_model: str,
    mode: str,
    target_prefix: str = DEFAULT_TARGET_PREFIX,
    training_metadata: dict[str, Any] | None = None,
) -> Path:
    """Save adapter-only scalars and the information needed to restore them."""
    from safetensors.torch import save_file

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    modules = list(named_moment_modules(model))
    if not modules:
        raise ValueError(
            "Cannot save an adapter from a model without MomentLinear modules."
        )

    module_names = [name for name, _ in modules]
    config = {
        "format": "moment-tuning",
        "format_version": FORMAT_VERSION,
        "base_model": base_model,
        "mode": mode,
        "target_prefix": target_prefix,
        "module_names": module_names,
        "module_count": len(module_names),
        "trainable_parameter_count": count_moment_parameters(model),
        "parameterization": "mu0 + sigma0 * mean_shift + exp(log_scale) * (W - mu0)",
        "training": training_metadata or {},
        "statistics": moment_statistics(model),
    }
    save_file(moment_state_dict(model), output_path / ADAPTER_WEIGHTS_NAME)
    (output_path / ADAPTER_CONFIG_NAME).write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def load_moment_adapter(
    model: nn.Module,
    adapter_dir: str | Path,
    *,
    expected_base_model: str | None = None,
) -> dict[str, Any]:
    """Inject MomentLinear modules and load a saved adapter in-place."""
    from safetensors.torch import load_file

    adapter_path = Path(adapter_dir)
    config = json.loads(
        (adapter_path / ADAPTER_CONFIG_NAME).read_text(encoding="utf-8")
    )
    if config.get("format") != "moment-tuning":
        raise ValueError(f"Unsupported adapter format: {config.get('format')!r}.")
    if config.get("format_version") != FORMAT_VERSION:
        raise ValueError(
            f"Unsupported adapter format version: {config.get('format_version')!r}."
        )
    if expected_base_model and config["base_model"] != expected_base_model:
        raise ValueError(
            f"Adapter expects {config['base_model']!r}, "
            f"but the requested base model is {expected_base_model!r}."
        )

    module_names = list(config["module_names"])
    inject_moment_adapters(
        model,
        mode=config["mode"],
        target_prefix=config["target_prefix"],
        module_names=module_names,
    )
    state = load_file(adapter_path / ADAPTER_WEIGHTS_NAME, device="cpu")
    expected_keys = {
        f"{name}.{parameter_name}"
        for name in module_names
        for parameter_name in ("mean_shift", "log_scale")
    }
    if set(state) != expected_keys:
        missing = expected_keys - set(state)
        unexpected = set(state) - expected_keys
        raise ValueError(
            f"Adapter state mismatch; missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}."
        )

    with torch.no_grad():
        modules = dict(named_moment_modules(model))
        for name in module_names:
            modules[name].mean_shift.copy_(state[f"{name}.mean_shift"])
            modules[name].log_scale.copy_(state[f"{name}.log_scale"])
    return config
