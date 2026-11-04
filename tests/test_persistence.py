"""Unit tests: run store round-trip, caching, secret scrubbing."""

import json
import unittest
from tempfile import TemporaryDirectory

from neev.persistence.runs import RunMeta, RunStore, sha256_json


class TestRunStore(unittest.TestCase):
    def test_save_and_load_roundtrip(self):
        with TemporaryDirectory() as tmp:
            store = RunStore(tmp)
            meta = RunMeta(purpose="smoke", models={"segment": "sam-3.1"})
            run_id = store.save({"a": 1}, {"b": 2}, meta)
            loaded = store.load(run_id)
            self.assertEqual(loaded["request"], {"a": 1})
            self.assertEqual(loaded["response"], {"b": 2})
            self.assertEqual(loaded["meta"]["models"], {"segment": "sam-3.1"})
            self.assertTrue(loaded["meta"]["created_utc"])

    def test_identical_request_hits_cache(self):
        with TemporaryDirectory() as tmp:
            store = RunStore(tmp)
            meta = RunMeta(purpose="smoke", models={})
            run_id = store.save({"a": 1}, {}, meta)
            self.assertEqual(store.find_cached({"a": 1}), run_id)
            self.assertIsNone(store.find_cached({"a": 2}))

    def test_secrets_are_scrubbed_from_meta(self):
        with TemporaryDirectory() as tmp:
            store = RunStore(tmp)
            meta = RunMeta(
                purpose="smoke",
                models={},
                notes={"api_key": "live-secret", "n": 1},
            )
            run_id = store.save({}, {}, meta)
            raw = (store.root / run_id / "meta.json").read_text(encoding="utf-8")
            self.assertNotIn("live-secret", raw)
            self.assertIn("[REDACTED]", raw)

    def test_unknown_run_raises_key_error(self):
        with TemporaryDirectory() as tmp:
            with self.assertRaises(KeyError):
                RunStore(tmp).load("nope")

    def test_request_hash_is_key_order_stable(self):
        self.assertEqual(sha256_json({"a": 1, "b": 2}), sha256_json({"b": 2, "a": 1}))


if __name__ == "__main__":
    unittest.main()
