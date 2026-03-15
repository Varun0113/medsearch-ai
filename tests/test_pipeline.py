import unittest
from pathlib import Path

from app.core.pipeline import MedSearchPipeline


INDEX_PATH = Path("indexes/medicine.index")


@unittest.skipUnless(INDEX_PATH.exists(), "medicine.index is required for semantic pipeline tests")
class MedSearchPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = MedSearchPipeline()

    def test_returns_medicines_for_disease_query(self):
        result = self.pipeline.run("diabetes", "en", {})

        self.assertTrue(result["conditions"])
        self.assertTrue(result["otc_medicines"] or result["prescription_medicines"])

    def test_combination_query_returns_conditions_and_treatments(self):
        result = self.pipeline.run("fever and cough", "en", {})

        self.assertTrue(result["conditions"])
        self.assertTrue(result["treatments"])

    def test_urgent_symptoms_override_medicine_output(self):
        result = self.pipeline.run("I have chest pain and trouble breathing", "en", {})

        self.assertEqual(result["otc_medicines"], [])
        self.assertEqual(result["prescription_medicines"], [])
        self.assertIn("URGENT MEDICAL ATTENTION REQUIRED", result["warnings"])

    def test_fallback_warning_is_returned_when_no_matches(self):
        result = self.pipeline.run("blurp flarn sketchy", "en", {})

        self.assertEqual(result["conditions"], [])
        self.assertIn(
            "No clear condition detected. Please provide more detailed symptoms.",
            result["warnings"],
        )


if __name__ == "__main__":
    unittest.main()
