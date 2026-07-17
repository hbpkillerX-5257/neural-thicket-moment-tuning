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


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(5, 7, bias=False),
                    nn.SiLU(),
                    nn.Linear(7, 5, bias=False),
                ),
                nn.Linear(5, 5, bias=True),
            ]
        )
        self.lm_head = nn.Linear(5, 9, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.model.language_model.layers[0](inputs)
        hidden = self.model.language_model.layers[1](hidden)
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

    def test_injection_targets_language_layers_only(self) -> None:
        model = TinyModel()
        names = inject_moment_adapters(model)

        self.assertEqual(len(names), 3)
        self.assertEqual(count_moment_parameters(model), 6)
        self.assertIsInstance(model.model.language_model.layers[0][0], MomentLinear)
        self.assertIsInstance(model.model.language_model.layers[0][2], MomentLinear)
        self.assertIsInstance(model.model.language_model.layers[1], MomentLinear)
        self.assertIsInstance(model.lm_head, nn.Linear)
        self.assertFalse(model.lm_head.weight.requires_grad)
        self.assertEqual(moment_statistics(model)["module_count"], 3)

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
