import json
import unittest

from annotation_service.prompt_normalization import (
    PromptRouteFailure,
    PromptTranslation,
    normalize_grounding_prompt,
)
from annotation_service.prompt_translation import (
    OpenAICompatiblePromptTranslator,
    PromptTranslationConfig,
    PromptTranslationError,
    parse_prompt_translation,
)


class FakeTranslator:
    def __init__(self, *, fail: bool = False):
        self.fail = fail
        self.calls: list[tuple[str, str]] = []

    def translate(self, prompt: str, *, profile: str):
        self.calls.append((prompt, profile))
        if self.fail:
            raise RuntimeError("synthetic translator failure")
        return PromptTranslation(
            translated_prompt=(
                "person without a helmet beside the blue equipment "
                "on the right"
            ),
            target_entities=("person",),
            preserved_constraints=(
                "without a helmet",
                "beside the blue equipment",
                "on the right",
            ),
            provider="fake-openai-compatible",
            model="fake-qwen",
            latency_ms=12.5,
        )


class FakeTransport:
    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, url, headers, body, timeout_seconds):
        self.calls.append(
            {
                "url": url,
                "headers": headers,
                "body": json.loads(body),
                "timeout_seconds": timeout_seconds,
            }
        )
        content = json.dumps(
            {
                "translated_prompt": "electrical distribution box",
                "target_entities": ["electrical distribution box"],
                "preserved_constraints": [],
                "warnings": [],
            }
        )
        return {"choices": [{"message": {"content": content}}]}


class PromptNormalizationOpenSemanticTest(unittest.TestCase):
    def test_known_direct_targets_bypass_llm(self):
        translator = FakeTranslator()

        result = normalize_grounding_prompt(
            "安全帽、反光背心、工人",
            mode="llm_grounding_caption",
            profile="open_semantic_zh_en_v1",
            translator=translator,
        )

        self.assertEqual(
            result.normalized_prompt,
            "helmet . safety vest . person .",
        )
        self.assertEqual(
            result.translation_provider,
            "deterministic-direct-targets",
        )
        self.assertEqual(
            result.translation_target_entities,
            ("helmet", "safety vest", "person"),
        )
        self.assertEqual(
            result.as_route(),
            {
                "rule_attempted": True,
                "rule_matched": True,
                "llm_attempted": False,
                "llm_succeeded": False,
                "fallback_used": False,
            },
        )
        self.assertEqual(translator.calls, [])

    def test_open_query_uses_llm_translation(self):
        translator = FakeTranslator()

        result = normalize_grounding_prompt(
            "找出画面右侧蓝色设备旁边没有佩戴安全帽的人员",
            mode="llm_grounding_caption",
            profile="open_semantic_zh_en_v1",
            translator=translator,
        )

        self.assertEqual(
            result.normalized_prompt,
            (
                "person without a helmet beside the blue equipment "
                "on the right ."
            ),
        )
        self.assertEqual(result.translation_model, "fake-qwen")
        self.assertFalse(result.translation_fallback_used)
        self.assertEqual(
            result.as_route(),
            {
                "rule_attempted": True,
                "rule_matched": False,
                "llm_attempted": True,
                "llm_succeeded": True,
                "fallback_used": False,
            },
        )
        self.assertEqual(len(translator.calls), 1)

    def test_translation_failure_can_fallback_or_fail_job(self):
        fallback = normalize_grounding_prompt(
            "蓝色设备旁的安全帽",
            mode="llm_grounding_caption",
            profile="open_semantic_zh_en_v1",
            translator=FakeTranslator(fail=True),
            translation_failure_policy="fallback_canonical_terms",
        )
        self.assertEqual(
            fallback.normalized_prompt,
            "蓝色设备旁的helmet .",
        )
        self.assertTrue(fallback.translation_fallback_used)
        self.assertEqual(
            fallback.translation_fallback_mode,
            "canonical_terms",
        )
        self.assertEqual(
            fallback.as_route(),
            {
                "rule_attempted": True,
                "rule_matched": False,
                "llm_attempted": True,
                "llm_succeeded": False,
                "fallback_used": True,
            },
        )

        with self.assertRaises(PromptRouteFailure) as raised:
            normalize_grounding_prompt(
                "蓝色设备旁的安全帽",
                mode="llm_grounding_caption",
                profile="open_semantic_zh_en_v1",
                translator=FakeTranslator(fail=True),
                translation_failure_policy="fail_job",
            )
        self.assertEqual(
            raised.exception.route,
            {
                "rule_attempted": True,
                "rule_matched": False,
                "llm_attempted": True,
                "llm_succeeded": False,
                "fallback_used": False,
            },
        )


class PromptTranslationProviderTest(unittest.TestCase):
    def test_strict_contract_and_process_cache(self):
        transport = FakeTransport()
        translator = OpenAICompatiblePromptTranslator(
            PromptTranslationConfig(
                base_url="http://qwen.example/v1",
                model="qwen25vl",
                api_key="secret",
            ),
            transport=transport,
        )

        first = translator.translate(
            "配电箱",
            profile="open_semantic_zh_en_v1",
        )
        second = translator.translate(
            "配电箱",
            profile="open_semantic_zh_en_v1",
        )

        self.assertEqual(
            first.translated_prompt,
            "electrical distribution box",
        )
        self.assertFalse(first.cache_hit)
        self.assertTrue(second.cache_hit)
        self.assertEqual(len(transport.calls), 1)
        self.assertEqual(
            transport.calls[0]["body"]["response_format"],
            {"type": "json_object"},
        )
        self.assertEqual(
            transport.calls[0]["headers"]["Authorization"],
            "Bearer secret",
        )

    def test_contract_rejects_extra_fields_and_untranslated_cjk(self):
        with self.assertRaises(PromptTranslationError):
            parse_prompt_translation(
                json.dumps(
                    {
                        "translated_prompt": "worker",
                        "target_entities": ["worker"],
                        "preserved_constraints": [],
                        "warnings": [],
                        "extra": "not allowed",
                    }
                )
            )
        with self.assertRaises(PromptTranslationError):
            parse_prompt_translation(
                json.dumps(
                    {
                        "translated_prompt": "施工人员",
                        "target_entities": ["person"],
                        "preserved_constraints": [],
                        "warnings": [],
                    },
                    ensure_ascii=False,
                )
            )


if __name__ == "__main__":
    unittest.main()
