import unittest
from fractions import Fraction

from evaluate_gsm8k import (
    canonical_number,
    extract_gold,
    extract_prediction,
)


class NumberParsingTests(unittest.TestCase):
    def test_canonical_number(self) -> None:
        self.assertEqual(canonical_number("1,250"), Fraction(1250))
        self.assertEqual(canonical_number("-3.5"), Fraction(-7, 2))
        self.assertEqual(canonical_number("7/2"), Fraction(7, 2))
        self.assertIsNone(canonical_number("not a number"))

    def test_extract_gold(self) -> None:
        raw, value = extract_gold("Reasoning with 3 and 4.\n#### 1,234")
        self.assertEqual(raw, "1,234")
        self.assertEqual(value, Fraction(1234))

    def test_marker_has_priority(self) -> None:
        raw, value, parser = extract_prediction(
            "I used 20 and 4.\n#### 5\nA trailing unrelated number is 99."
        )
        self.assertEqual(raw, "5")
        self.assertEqual(value, Fraction(5))
        self.assertEqual(parser, "####")

    def test_boxed_fallback(self) -> None:
        raw, value, parser = extract_prediction(r"Thus the result is \boxed{-2.5}.")
        self.assertEqual(raw, "-2.5")
        self.assertEqual(value, Fraction(-5, 2))
        self.assertEqual(parser, "boxed")

    def test_last_number_fallback(self) -> None:
        raw, value, parser = extract_prediction("After calculating, I get 42.")
        self.assertEqual(raw, "42")
        self.assertEqual(value, Fraction(42))
        self.assertEqual(parser, "last-number")

    def test_unparsed(self) -> None:
        raw, value, parser = extract_prediction("I cannot solve this.")
        self.assertIsNone(raw)
        self.assertIsNone(value)
        self.assertEqual(parser, "unparsed")


if __name__ == "__main__":
    unittest.main()
