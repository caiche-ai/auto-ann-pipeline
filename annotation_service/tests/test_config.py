import os
import unittest
from unittest.mock import patch

from annotation_service.config import Settings


class SettingsTest(unittest.TestCase):
    def test_defaults(self):
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings.from_env()

        self.assertEqual(settings.service_version, "1.5.1")
        self.assertIsNone(settings.api_key)
        self.assertEqual(settings.cors_origins, ())
        self.assertFalse(settings.cors_allow_credentials)
        self.assertEqual(settings.max_request_bytes, 30 * 1024 * 1024)
        self.assertEqual(settings.max_image_bytes, 20 * 1024 * 1024)
        self.assertEqual(settings.max_queued_jobs, 100)
        self.assertTrue(settings.docs_enabled)
        self.assertFalse(settings.storage_enabled)
        self.assertEqual(settings.storage_root, "./annotation-data")
        self.assertEqual(
            settings.prompt_normalization_mode,
            "terminal_period",
        )
        self.assertEqual(
            settings.prompt_normalization_profile,
            "construction_safety_v1",
        )
        self.assertEqual(
            settings.prompt_translation_failure_policy,
            "fallback_canonical_terms",
        )

    def test_open_semantic_environment_values_are_validated(self):
        env = {
            "ANNOTATION_GROUNDING_DINO_PROMPT_NORMALIZATION_MODE": (
                "llm_grounding_caption"
            ),
            "ANNOTATION_GROUNDING_DINO_PROMPT_NORMALIZATION_PROFILE": (
                "open_semantic_zh_en_v1"
            ),
            "ANNOTATION_GROUNDING_DINO_PROMPT_TRANSLATION_FAILURE_POLICY": (
                "fail_job"
            ),
        }
        with patch.dict(os.environ, env, clear=True):
            settings = Settings.from_env()
        self.assertEqual(
            settings.prompt_normalization_mode,
            "llm_grounding_caption",
        )
        self.assertEqual(
            settings.prompt_translation_failure_policy,
            "fail_job",
        )

    def test_environment_values_are_parsed(self):
        env = {
            "ANNOTATION_SERVICE_VERSION": "1.2.3",
            "ANNOTATION_API_KEY": " secret ",
            "ANNOTATION_CORS_ORIGINS": (
                "http://localhost:3000/,https://annotation.example.com"
            ),
            "ANNOTATION_DOCS_ENABLED": "false",
            "ANNOTATION_MAX_QUEUED_JOBS": "25",
            "ANNOTATION_GROUNDING_DINO_PROMPT_NORMALIZATION_MODE": "off",
            "ANNOTATION_GROUNDING_DINO_PROMPT_NORMALIZATION_PROFILE": (
                "construction_safety_v1"
            ),
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
        self.assertEqual(settings.prompt_normalization_mode, "off")
        self.assertEqual(
            settings.prompt_normalization_profile,
            "construction_safety_v1",
        )

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
