import json
import unittest

from annotation_service.qwen_contract import (
    QwenImageInput,
    QwenVisualContext,
)
from annotation_service.qwen_provider import (
    Qwen25VLProvider,
    QwenProviderConfig,
    QwenProviderError,
)


FACTS = {
    "target_object": "画面中央的一名作业人员",
    "instance_count": 1,
    "visual_anchor": ["位于画面中央"],
    "mask_granularity": "人员整体",
    "visible_facts": ["人员头部未见安全帽"],
    "risk_semantics": "头部防护缺失",
}
PROMPTS = {
    "prompts": [
        {"prompt_id": "visual-1", "type": "visual", "text": "分割画面中央的人员。"},
        {"prompt_id": "visual-2", "type": "visual", "text": "标出中央位置的作业人员。"},
        {"prompt_id": "visual-3", "type": "visual", "text": "提取图像中部的目标人员。"},
        {"prompt_id": "risk-1", "type": "risk", "text": "分割中央未戴安全帽的人员。"},
        {"prompt_id": "risk-2", "type": "risk", "text": "标出中央头部防护缺失人员。"},
        {"prompt_id": "agent-1", "type": "agent", "text": "请定位并分割中央未戴安全帽的人员。"},
    ]
}


class FakeTransport:
    def __init__(self):
        self.calls = []

    def __call__(self, url, headers, body, timeout):
        payload = json.loads(body)
        self.calls.append((url, headers, payload, timeout))
        content = FACTS if len(self.calls) == 1 else PROMPTS
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            content,
                            ensure_ascii=False,
                        )
                    }
                }
            ]
        }


class QwenProviderTest(unittest.TestCase):
    def test_two_stage_generation_uses_one_configured_model(self):
        transport = FakeTransport()
        provider = Qwen25VLProvider(
            QwenProviderConfig(
                base_url="http://qwen25vl:8000/v1",
                model="qwen2.5-vl-7b-instruct",
                api_key="secret",
            ),
            transport=transport,
        )
        result = provider.generate(
            context=QwenVisualContext(
                asset_id="asset-1",
                category="helmet_missing",
                target_box_xyxy=[1, 1, 9, 9],
                mask_available=True,
            ),
            images=[
                QwenImageInput(
                    label="原图",
                    media_type="image/png",
                    data_url="data:image/png;base64,aW1hZ2U=",
                )
            ],
        )

        self.assertEqual(len(transport.calls), 2)
        self.assertEqual(
            transport.calls[0][0],
            "http://qwen25vl:8000/v1/chat/completions",
        )
        self.assertEqual(
            transport.calls[0][1]["Authorization"],
            "Bearer secret",
        )
        self.assertEqual(
            transport.calls[0][2]["model"],
            "qwen2.5-vl-7b-instruct",
        )
        first_content = transport.calls[0][2]["messages"][1]["content"]
        self.assertTrue(
            any(item["type"] == "image_url" for item in first_content)
        )
        second_content = transport.calls[1][2]["messages"][1]["content"]
        self.assertIsInstance(second_content, str)
        self.assertEqual(len(result.prompt_set.prompts), 6)
        self.assertEqual(
            result.as_dict()["provenance"]["qwen_model"],
            "qwen2.5-vl-7b-instruct",
        )

    def test_missing_assistant_content_is_rejected(self):
        provider = Qwen25VLProvider(
            QwenProviderConfig(base_url="http://localhost:8000/v1"),
            transport=lambda *_: {"choices": []},
        )
        with self.assertRaises(QwenProviderError):
            provider.generate(
                context=QwenVisualContext(
                    asset_id="asset-1",
                    category="safe",
                    target_box_xyxy=[1, 1, 2, 2],
                ),
                images=[
                    QwenImageInput(
                        label="原图",
                        media_type="image/png",
                        data_url="data:image/png;base64,aQ==",
                    )
                ],
            )


if __name__ == "__main__":
    unittest.main()
