"""Unit tests: config, secrets handling. No network, no API calls."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from neev.config import (
    credentials_status,
    load_config,
    parse_dotenv,
    redact,
)


class TestDotenv(unittest.TestCase):
    def test_parses_values_skips_comments_and_blanks(self):
        text = "# comment\nMODEL_API_KEY=abc123\n\nEMPTY=\nQUOTED=\"x y\"\n"
        self.assertEqual(
            parse_dotenv(text),
            {"MODEL_API_KEY": "abc123", "EMPTY": "", "QUOTED": "x y"},
        )


class TestLoadConfig(unittest.TestCase):
    def test_env_wins_over_dotenv(self):
        with TemporaryDirectory() as tmp:
            dotenv = Path(tmp) / ".env"
            dotenv.write_text("MODEL_API_KEY=file-key\n", encoding="utf-8")
            config = load_config(
                env={"MODEL_API_KEY": "env-key"}, dotenv_path=dotenv
            )
            self.assertEqual(config.model_api_key, "env-key")

    def test_dotenv_fills_gaps(self):
        with TemporaryDirectory() as tmp:
            dotenv = Path(tmp) / ".env"
            dotenv.write_text("MAPILLARY_TOKEN=tok\n", encoding="utf-8")
            config = load_config(env={}, dotenv_path=dotenv)
            self.assertEqual(config.mapillary_token, "tok")
            self.assertIsNone(config.model_api_key)

    def test_missing_everything_stays_none(self):
        with TemporaryDirectory() as tmp:
            config = load_config(env={}, dotenv_path=Path(tmp) / ".env")
            self.assertIsNone(config.model_api_key)
            self.assertIsNone(config.mapillary_token)


class TestSecrets(unittest.TestCase):
    def test_status_reports_presence_only(self):
        config = load_config(env={"MODEL_API_KEY": "super-secret-value"})
        status = credentials_status(config)
        self.assertEqual(
            status, {"MODEL_API_KEY": True, "MAPILLARY_TOKEN": False}
        )
        self.assertNotIn("super-secret-value", str(status))

    def test_redact_scrubs_secret_keys_recursively(self):
        dirty = {
            "api_key": "abc",
            "nested": [{"token": "def", "ok": 1}],
            "note": "keep",
        }
        self.assertEqual(
            redact(dirty),
            {
                "api_key": "[REDACTED]",
                "nested": [{"token": "[REDACTED]", "ok": 1}],
                "note": "keep",
            },
        )


if __name__ == "__main__":
    unittest.main()
