import unittest

import torch

from train_gsm8k_moments import CausalLMCollator, tokenize_example


class FakeTokenizer:
    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        enable_thinking,
    ):
        assert not tokenize
        assert not enable_thinking
        prompt = "prompt"
        if add_generation_prompt:
            return prompt
        assert messages[-1]["role"] == "assistant"
        return prompt + "answer"

    def __call__(self, text, *, add_special_tokens):
        assert not add_special_tokens
        if text == "prompt":
            return {"input_ids": [10, 11, 12, 13]}
        if text == "promptanswer":
            return {"input_ids": [10, 11, 12, 13, 21, 22, 23, 24]}
        raise AssertionError(f"Unexpected text: {text!r}")


class TrainingUtilityTests(unittest.TestCase):
    def test_response_only_labels_mask_prompt(self) -> None:
        encoded = tokenize_example(
            {"question": "What is 1+1?", "answer": "2\n#### 2"},
            tokenizer=FakeTokenizer(),
            max_length=16,
        )

        self.assertEqual(encoded["input_ids"], [10, 11, 12, 13, 21, 22, 23, 24])
        self.assertEqual(encoded["labels"], [-100, -100, -100, -100, 21, 22, 23, 24])
        self.assertFalse(encoded["truncated"])

    def test_truncation_preserves_response_only_boundary(self) -> None:
        encoded = tokenize_example(
            {"question": "What is 1+1?", "answer": "2\n#### 2"},
            tokenizer=FakeTokenizer(),
            max_length=6,
        )

        self.assertEqual(encoded["input_ids"], [10, 11, 12, 13, 21, 22])
        self.assertEqual(encoded["labels"], [-100, -100, -100, -100, 21, 22])
        self.assertTrue(encoded["truncated"])

    def test_collator_right_pads_and_masks_labels(self) -> None:
        collator = CausalLMCollator(pad_token_id=0)
        batch = collator(
            [
                {
                    "input_ids": [1, 2, 3],
                    "attention_mask": [1, 1, 1],
                    "labels": [-100, 2, 3],
                },
                {
                    "input_ids": [4, 5],
                    "attention_mask": [1, 1],
                    "labels": [-100, 5],
                },
            ]
        )

        torch.testing.assert_close(
            batch["input_ids"],
            torch.tensor([[1, 2, 3], [4, 5, 0]]),
        )
        torch.testing.assert_close(
            batch["attention_mask"],
            torch.tensor([[1, 1, 1], [1, 1, 0]]),
        )
        torch.testing.assert_close(
            batch["labels"],
            torch.tensor([[-100, 2, 3], [-100, 5, -100]]),
        )


if __name__ == "__main__":
    unittest.main()
