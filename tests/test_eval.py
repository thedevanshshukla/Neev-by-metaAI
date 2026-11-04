"""Unit tests: label format and metrics with unclear handling."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from neev.eval.labels import load_labels, validate_labels
from neev.eval.metrics import (
    agreement,
    binary_pr,
    consistency,
    exact_match_rate,
)


def labels_doc(**overrides):
    doc = {
        "version": 1,
        "dataset": "t",
        "buildings": [
            {
                "building_id": "B1",
                "image": "f_0001.jpg",
                "labels": {"open_ground_storey": "yes", "storeys": 4},
            }
        ],
    }
    doc.update(overrides)
    return doc


class TestLabels(unittest.TestCase):
    def test_valid_labels_pass(self):
        self.assertEqual(validate_labels(labels_doc()), [])

    def test_example_file_loads(self):
        example = Path(__file__).resolve().parent.parent / "eval" / "example_labels.json"
        doc = load_labels(example)
        self.assertEqual(len(doc["buildings"]), 2)

    def test_bad_version_fails(self):
        self.assertTrue(validate_labels(labels_doc(version=99)))

    def test_unknown_label_fails(self):
        doc = labels_doc()
        doc["buildings"][0]["labels"]["bogus"] = "yes"
        self.assertTrue(validate_labels(doc))

    def test_bad_storeys_fails(self):
        doc = labels_doc()
        doc["buildings"][0]["labels"]["storeys"] = 0
        self.assertTrue(validate_labels(doc))

    def test_load_lists_every_problem(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "labels.json"
            path.write_text(json.dumps({"version": 1}), encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                load_labels(path)
            self.assertIn("buildings", str(ctx.exception))


class TestMetrics(unittest.TestCase):
    def test_binary_pr_excludes_unclear_by_default(self):
        pairs = [("yes", "yes"), ("no", "no"), ("unclear", "yes"), ("yes", "no")]
        result = binary_pr(pairs)
        self.assertEqual((result.tp, result.fp, result.fn, result.tn), (1, 1, 0, 1))
        self.assertEqual(result.unclear, 1)
        self.assertAlmostEqual(result.precision, 0.5)
        self.assertAlmostEqual(result.recall, 1.0)
        self.assertAlmostEqual(result.unclear_rate, 0.25)

    def test_binary_pr_as_error_counts_unclear_wrong(self):
        pairs = [("yes", "yes"), ("unclear", "yes")]
        result = binary_pr(pairs, unclear_mode="as_error")
        self.assertEqual((result.tp, result.fn), (1, 1))
        self.assertAlmostEqual(result.recall, 0.5)

    def test_zero_denominator_is_none_not_zero(self):
        result = binary_pr([("no", "no")])
        self.assertIsNone(result.precision)
        self.assertIsNone(result.recall)

    def test_agreement_and_exact_match(self):
        self.assertAlmostEqual(agreement([("a", "a"), ("a", "b")]).rate, 0.5)
        self.assertAlmostEqual(exact_match_rate([(4, 4), (4, 5)]).rate, 0.5)

    def test_consistency_requires_aligned_runs(self):
        with self.assertRaises(ValueError):
            consistency(["yes"], ["yes", "no"])
        self.assertAlmostEqual(consistency(["yes", "no"], ["yes", "no"]).rate, 1.0)


if __name__ == "__main__":
    unittest.main()
