import unittest

from torch import nn

from train_gsm8k_lora import LORA_TARGET_SUFFIXES, discover_lora_targets


class TinyLanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "mlp": nn.ModuleDict(
                            {
                                "gate_proj": nn.Linear(4, 8, bias=False),
                                "up_proj": nn.Linear(4, 8, bias=False),
                                "down_proj": nn.Linear(8, 4, bias=False),
                            }
                        )
                    }
                )
            ]
        )
        # Should be ignored: outside language layers.
        self.visual = nn.Linear(4, 4, bias=False)
        self.lm_head = nn.Linear(4, 9, bias=False)


class LoRATargetDiscoveryTests(unittest.TestCase):
    def test_discovers_language_layer_linears_only(self) -> None:
        model = TinyLanguageModel()
        targets = discover_lora_targets(model)
        self.assertEqual(len(targets), 3)
        self.assertTrue(
            all(name.startswith("model.language_model.layers.") for name in targets)
        )
        self.assertTrue(
            all(
                any(name.endswith(suffix) for suffix in LORA_TARGET_SUFFIXES)
                for name in targets
            )
        )
        self.assertNotIn("visual", " ".join(targets))
        self.assertNotIn("lm_head", " ".join(targets))


if __name__ == "__main__":
    unittest.main()
