import unittest

from fastapi.testclient import TestClient

from annotation_service.api.app import create_app
from annotation_service.api.config import Settings
from annotation_service.pipeline.qwen.contract import (
    DEFAULT_QWEN_PROMPT_CONTEXT_PLACEHOLDER,
    DEFAULT_QWEN_PROMPT_TEMPLATE,
    DEFAULT_QWEN_PROMPT_TEMPLATE_ID,
)


class PromptTemplateApiTest(unittest.TestCase):
    def setUp(self):
        self.client_context = TestClient(
            create_app(
                Settings(
                    api_key="test-key",
                    docs_enabled=True,
                    storage_enabled=False,
                )
            )
        )
        self.client = self.client_context.__enter__()

    def tearDown(self):
        self.client_context.__exit__(None, None, None)

    def test_default_template_requires_authentication(self):
        response = self.client.get(
            "/v1/annotation/prompt-templates/default"
        )

        self.assertEqual(response.status_code, 401)

    def test_default_template_returns_complete_template(self):
        response = self.client.get(
            "/v1/annotation/prompt-templates/default",
            headers={"Authorization": "Bearer test-key"},
        )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(
            response.json(),
            {
                "template_id": DEFAULT_QWEN_PROMPT_TEMPLATE_ID,
                "template": DEFAULT_QWEN_PROMPT_TEMPLATE,
                "context_placeholder": (
                    DEFAULT_QWEN_PROMPT_CONTEXT_PLACEHOLDER
                ),
            },
        )


if __name__ == "__main__":
    unittest.main()
