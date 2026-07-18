import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from moment_tuning import (
    MomentLinear,
    _MomentAffine,
    count_moment_parameters,
    inject_moment_adapters,
    load_moment_adapter,
    moment_statistics,
    save_moment_adapter,
)


class TinyLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(5, 5, bias=False)
        self.self_attn.o_proj = nn.Linear(5, 5, bias=False)
        self.mlp = nn.Module()
        self.mlp.gate_proj = nn.Linear(5, 7, bias=False)
        self.mlp.up_proj = nn.Linear(5, 7, bias=False)
        self.mlp.down_proj = nn.Linear(7, 5, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.self_attn.o_proj(self.self_attn.q_proj(inputs))
        gated = F.silu(self.mlp.gate_proj(hidden)) * self.mlp.up_proj(hidden)
        return self.mlp.down_proj(gated)


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.layers = nn.ModuleList([TinyLayer(), TinyLayer()])
        self.lm_head = nn.Linear(5, 9, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = inputs
        for layer in self.model.language_model.layers:
            hidden = layer(hidden)
        return self.lm_head(hidden)


class MomentLinearTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)

    def test_initialization_exactly_matches_frozen_linear(self) -> None:
        linear = nn.Linear(5, 4, bias=True)
        inputs = torch.randn(3, 2, 5)
        expected = linear(inputs)

        adapted = MomentLinear(linear)
        actual = adapted(inputs)

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_efficient_forward_matches_materialized_weight(self) -> None:
        linear = nn.Linear(5, 4, bias=True)
        adapted = MomentLinear(linear)
        with torch.no_grad():
            adapted.mean_shift.fill_(0.17)
            adapted.log_scale.fill_(-0.08)

        inputs = torch.randn(3, 2, 5)
        expected = F.linear(inputs, adapted.materialized_weight(), adapted.bias)
        actual = adapted(inputs)

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_only_moment_scalars_receive_parameter_gradients(self) -> None:
        adapted = MomentLinear(nn.Linear(5, 4, bias=True))
        inputs = torch.randn(3, 5)
        adapted(inputs).square().mean().backward()

        self.assertIsNotNone(adapted.mean_shift.grad)
        self.assertIsNotNone(adapted.log_scale.grad)
        self.assertIsNone(adapted.weight.grad)
        self.assertIsNone(adapted.bias.grad)
        trainable = [
            name
            for name, parameter in adapted.named_parameters()
            if parameter.requires_grad
        ]
        self.assertEqual(trainable, ["mean_shift", "log_scale"])

    def test_scalar_gradients_reduce_in_fp32(self) -> None:
        base = torch.ones(100_000, dtype=torch.float16)
        input_sum = torch.ones(100_000, dtype=torch.float32)
        scale = torch.tensor(1.0, dtype=torch.float32, requires_grad=True)
        shift = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

        output = _MomentAffine.apply(base, input_sum, scale, shift)
        output.sum().backward()

        self.assertTrue(torch.isfinite(scale.grad))
        self.assertTrue(torch.isfinite(shift.grad))
        self.assertEqual(scale.grad.item(), 100_000)
        self.assertEqual(shift.grad.item(), 100_000)

    def test_injection_defaults_to_mlp_only(self) -> None:
        model = TinyModel()
        names = inject_moment_adapters(model)

        self.assertEqual(len(names), 6)
        self.assertTrue(all(".mlp." in name for name in names))
        self.assertFalse(any("self_attn" in name for name in names))
        self.assertEqual(count_moment_parameters(model), 12)
        layer = model.model.language_model.layers[0]
        self.assertIsInstance(layer.mlp.gate_proj, MomentLinear)
        self.assertIsInstance(layer.mlp.up_proj, MomentLinear)
        self.assertIsInstance(layer.mlp.down_proj, MomentLinear)
        self.assertIsInstance(layer.self_attn.q_proj, nn.Linear)
        self.assertIsInstance(layer.self_attn.o_proj, nn.Linear)
        self.assertIsInstance(model.lm_head, nn.Linear)
        self.assertFalse(model.lm_head.weight.requires_grad)
        self.assertEqual(moment_statistics(model)["module_count"], 6)

    def test_injection_can_target_attention_or_all(self) -> None:
        attention_model = TinyModel()
        attention_names = inject_moment_adapters(
            attention_model, target_scope="attention"
        )
        self.assertEqual(len(attention_names), 4)
        self.assertTrue(all("self_attn" in name for name in attention_names))

        all_model = TinyModel()
        all_names = inject_moment_adapters(all_model, target_scope="all")
        self.assertEqual(len(all_names), 10)

    def test_adapter_round_trip_is_lossless(self) -> None:
        model = TinyModel()
        inject_moment_adapters(model)
        with torch.no_grad():
            for index, module in enumerate(
                module for module in model.modules() if isinstance(module, MomentLinear)
            ):
                module.mean_shift.fill_(0.01 * (index + 1))
                module.log_scale.fill_(-0.02 * (index + 1))

        inputs = torch.randn(2, 5)
        expected = model(inputs)

        with tempfile.TemporaryDirectory() as directory:
            save_moment_adapter(
                model,
                directory,
                base_model="tiny/test",
                mode="both",
                target_scope="mlp",
                training_metadata={"step": 3},
            )
            restored = TinyModel()
            restored.load_state_dict(TinyModel().state_dict())

            # Match the frozen base weights before applying adapter scalars.
            base_state = {
                key.replace(".weight", ".weight"): value
                for key, value in model.state_dict().items()
                if "mean_shift" not in key
                and "log_scale" not in key
                and "pretrained_mean" not in key
                and "pretrained_std" not in key
            }
            restored.load_state_dict(base_state, strict=False)
            config = load_moment_adapter(
                restored,
                Path(directory),
                expected_base_model="tiny/test",
            )
            actual = restored(inputs)

        self.assertEqual(config["training"]["step"], 3)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
