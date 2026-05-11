"""Unit tests for German text normalization."""

from __future__ import annotations

import unittest

from kokorotts.de_normalizer import int_to_de, normalize_text_de, ordinal_stem_de, year_de


class GermanNormalizerTest(unittest.TestCase):
    def test_int_to_de_formats_common_numbers(self) -> None:
        self.assertEqual(int_to_de(0), "null")
        self.assertEqual(int_to_de(42), "zweiundvierzig")
        self.assertEqual(int_to_de(1_000), "eintausend")

    def test_ordinal_stem_de_formats_common_ordinals(self) -> None:
        self.assertEqual(ordinal_stem_de(1), "erst")
        self.assertEqual(ordinal_stem_de(3), "dritt")
        self.assertEqual(ordinal_stem_de(21), "einundzwanzigst")

    def test_year_de_formats_historical_years(self) -> None:
        self.assertEqual(year_de(1989), "neunzehnhundertneunundachtzig")
        self.assertEqual(year_de(2026), "zweitausendsechsundzwanzig")

    def test_normalize_text_de_expands_date_time_currency_and_abbreviation(self) -> None:
        normalized = normalize_text_de("Dr. Müller zahlt 1.299,99€ am 03.04.2026 um 14:30.")

        self.assertIn("Doktor Müller", normalized)
        self.assertIn("eintausendzweihundertneunundneunzig Euro", normalized)
        self.assertIn("neunundneunzig Cent", normalized)
        self.assertIn("dritte April zweitausendsechsundzwanzig", normalized)
        self.assertIn("vierzehn Uhr dreißig", normalized)
        self.assertNotIn("€", normalized)


if __name__ == "__main__":
    unittest.main()
