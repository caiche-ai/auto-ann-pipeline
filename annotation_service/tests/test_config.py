import os
import unittest
from unittest.mock import patch

from annotation_service.config import Settings


class SettingsTest(unittest.TestCase):
    def test_defaults(self):
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings.from_env()

        self.assertEqual(settings.service_version, "1.1.0")
        self.assertIsNone(settings.api_key)
        self.assertEqual(settings.cors_origins, ())
        self.assertFalse(settings.cors_allow_credentials)
        self.assertEqual(settings.max_request_bytes, 30 * 1024 * 1024)
        self.assertEqual(settings.max_image_bytes, 20 * 1024 * 1024)
        self.assertEqual(settings.max_queued_jobs, 100)
        self.assertTrue(settings.docs_enabled)
        self.assertFalse(settings.storage_enabled)
        self.assertEqual(settings.storage_root, "./annotation-data")

    def test_environment_values_are_parsed(self):
        env = {
            "ANNOTATION_SERVICE_VERSION": "1.2.3",
            "ANNOTATION_API_KEY": " secret ",
            "ANNOTATION_CORS_ORIGINS": (
                "http://localhost:3000/,https://annotation.example.com"
            ),
            "ANNOTATION_DOCS_ENABLED": "false",
            "ANNOTATION_MAX_QUEUED_JOBS": "25",
        }
        with patch.dict(os.environ, env, clear=True):
            settings = Settings.from_env()

        self.assertEqual(settings.service_version, "1.2.3")
        self.assertEqual(settings.api_key, "secret")
        self.assertEqual(
            settings.cors_origins,
            (
                "http://localhost:3000",
                "https://annotation.example.com",
            ),
        )
        self.assertFalse(settings.docs_enabled)
        self.assertEqual(settings.max_queued_jobs, 25)

    def test_request_limit_must_exceed_image_limit(self):
        with patch.dict(
            os.environ,
            {
                "ANNOTATION_MAX_REQUEST_BYTES": "100",
                "ANNOTATION_MAX_IMAGE_BYTES": "100",
            },
            clear=True,
        ):
            with self.assertRaises(ValueError):
                Settings.from_env()

    def test_credentialed_cors_rejects_wildcard(self):
        with patch.dict(
            os.environ,
            {
                "ANNOTATION_CORS_ORIGINS": "*",
                "ANNOTATION_CORS_ALLOW_CREDENTIALS": "true",
            },
            clear=True,
        ):
            with self.assertRaises(ValueError):
                Settings.from_env()

    def test_invalid_boolean_is_rejected(self):
        with patch.dict(
            os.environ,
            {"ANNOTATION_DOCS_ENABLED": "sometimes"},
            clear=True,
        ):
            with self.assertRaises(ValueError):
                Settings.from_env()

    def test_enabled_storage_requires_root(self):
        with patch.dict(
            os.environ,
            {
                "ANNOTATION_STORAGE_ENABLED": "true",
                "ANNOTATION_STORAGE_ROOT": "   ",
            },
            clear=True,
        ):
            with self.assertRaises(ValueError):
                Settings.from_env()


if __name__ == "__main__":
    unittest.main()
