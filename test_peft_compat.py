import unittest
from unittest import mock

import peft.import_utils as peft_import_utils

from peft_compat import disable_incompatible_torchao


class TorchAOCompatTests(unittest.TestCase):
    def test_disables_old_torchao_across_peft_modules(self) -> None:
        original = peft_import_utils.is_torchao_available
        fake_module = mock.Mock()
        fake_module.is_torchao_available = original
        try:
            with (
                mock.patch(
                    "peft_compat._torchao_version",
                    return_value="0.10.0",
                ),
                mock.patch.dict(
                    "sys.modules",
                    {"peft.tuners.lora.torchao": fake_module},
                ),
            ):
                disabled = disable_incompatible_torchao()
            self.assertTrue(disabled)
            self.assertFalse(peft_import_utils.is_torchao_available())
            self.assertFalse(fake_module.is_torchao_available())
        finally:
            peft_import_utils.is_torchao_available = original

    def test_leaves_new_torchao_alone(self) -> None:
        original = peft_import_utils.is_torchao_available
        try:
            with mock.patch(
                "peft_compat._torchao_version",
                return_value="0.16.0",
            ):
                disabled = disable_incompatible_torchao()
            self.assertFalse(disabled)
            self.assertIs(peft_import_utils.is_torchao_available, original)
        finally:
            peft_import_utils.is_torchao_available = original

    def test_noop_when_torchao_missing(self) -> None:
        with mock.patch("peft_compat._torchao_version", return_value=None):
            self.assertFalse(disable_incompatible_torchao())


if __name__ == "__main__":
    unittest.main()
