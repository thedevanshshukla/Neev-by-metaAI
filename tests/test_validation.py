"""Unit tests: output validation, Re-film rule, scoring (docs.md 8)."""

import unittest

from neev.validation.schemas import missing_ids, validate_building
from neev.validation.uncertainty import (
    needs_refilm,
    score_building,
    storey_crosscheck,
    storey_points,
)


def item(iid, answer, evidence="f_0001.jpg", confidence=0.9, reason="seen"):
    return {
        "id": iid,
        "answer": answer,
        "evidence": evidence,
        "confidence": confidence,
        "reason": reason,
    }


def building(**overrides):
    doc = {
        "building_id": "LN-B1",
        "best_frames": ["f_0001.jpg", "f_0002.jpg"],
        "storeys": {"label": "G+3", "measured": 4, "judge": 4},
        "checklist": [
            item("open_ground_storey", "yes"),
            item("heavy_overhang", "yes", evidence="f_0002.jpg"),
            item("pounding", "no"),
            {
                "id": "short_column",
                "answer": "unclear",
                "evidence": None,
                "confidence": None,
                "reason": "hidden by cars",
            },
        ],
        "score": 55,
        "band": "high",
        "models": {"segment": "sam-3.1", "judge": "muse-wonder-1.0"},
    }
    doc.update(overrides)
    return doc


class TestValidation(unittest.TestCase):
    def test_valid_building_passes(self):
        self.assertEqual(validate_building(building()), [])

    def test_yes_without_evidence_fails(self):
        doc = building()
        doc["checklist"][0]["evidence"] = "f_9999.jpg"
        self.assertTrue(validate_building(doc))

    def test_yes_without_reason_fails(self):
        doc = building()
        doc["checklist"][0]["reason"] = "  "
        self.assertTrue(validate_building(doc))

    def test_unclear_with_evidence_fails(self):
        doc = building()
        doc["checklist"][3]["evidence"] = "f_0001.jpg"
        self.assertTrue(validate_building(doc))

    def test_unknown_id_fails(self):
        doc = building()
        doc["checklist"][0]["id"] = "made_up"
        self.assertTrue(validate_building(doc))

    def test_missing_ids_lists_gaps(self):
        gaps = missing_ids(building())
        self.assertIn("falling_hazards", gaps)
        self.assertNotIn("pounding", gaps)


class TestUncertainty(unittest.TestCase):
    def test_crosscheck_disagreement_is_unclear(self):
        self.assertIsNone(storey_crosscheck(4, 5))
        self.assertEqual(storey_crosscheck(4, 4), 4)

    def test_storey_points(self):
        self.assertEqual(storey_points(6), 10)
        self.assertEqual(storey_points(4), 5)
        self.assertEqual(storey_points(2), 0)

    def test_refilm_on_unclear_soft_storey(self):
        self.assertTrue(needs_refilm({"open_ground_storey": "unclear"}))

    def test_refilm_on_three_unclear(self):
        answers = {
            "open_ground_storey": "no",
            "heavy_overhang": "unclear",
            "pounding": "unclear",
            "short_column": "unclear",
        }
        self.assertTrue(needs_refilm(answers))

    def test_docs_example_b1_scores_55_high(self):
        answers = {
            "open_ground_storey": "yes",
            "heavy_overhang": "yes",
            "pounding": "no",
            "falling_hazards": "no",
            "short_column": "no",
            "vertical_irregularity": "no",
            "apparent_quality": "no",
            "construction_type": "no",
        }
        result = score_building(answers, 4)
        self.assertEqual(result.score, 55)
        self.assertEqual(result.band, "high")
        self.assertFalse(result.refilm)

    def test_refilm_has_no_colour(self):
        result = score_building({"open_ground_storey": "unclear"}, 4)
        self.assertEqual(result.band, "re-film")
        self.assertTrue(result.refilm)


if __name__ == "__main__":
    unittest.main()
