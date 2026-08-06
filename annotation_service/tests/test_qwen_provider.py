import json
import unittest

from annotation_service.qwen_contract import (
    QwenContractError,
    QwenImageInput,
    QwenJointTarget,
    QwenJointVisualContext,
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
    def __init__(self, facts=None, prompts=None):
        self.calls = []
        self.facts = FACTS if facts is None else facts
        self.prompts = PROMPTS if prompts is None else prompts

    def __call__(self, url, headers, body, timeout):
        payload = json.loads(body)
        self.calls.append((url, headers, payload, timeout))
        content = self.facts if len(self.calls) == 1 else self.prompts
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

    def test_joint_generation_uses_joint_prompt_versions(self):
        task_ids = ["tsk-1", "tsk-2"]
        transport = FakeTransport(
            facts={
                **FACTS,
                "target_object": "一名人员和相邻设备",
                "instance_count": 2,
                "task_targets": [
                    {
                        "task_id": task_ids[0],
                        "target_object": "一名人员",
                        "instance_count": 1,
                        "visual_anchor": ["位于画面左侧"],
                    },
                    {
                        "task_id": task_ids[1],
                        "target_object": "一台挖掘机",
                        "instance_count": 1,
                        "visual_anchor": ["位于人员右侧"],
                    },
                ],
            },
            prompts={
                **PROMPTS,
                "covered_task_ids": task_ids,
                "fact_consistent": True,
            },
        )
        provider = Qwen25VLProvider(
            QwenProviderConfig(base_url="http://qwen25vl:8000/v1"),
            transport=transport,
        )

        result = provider.generate_joint(
            context=QwenJointVisualContext(
                asset_id="asset-1",
                targets=[
                    QwenJointTarget(
                        task_id="tsk-1",
                        task_version=1,
                        category="unsafe",
                        candidate_target_object="person",
                        target_box_xyxy=[1, 1, 4, 8],
                    ),
                    QwenJointTarget(
                        task_id="tsk-2",
                        task_version=1,
                        category="equipment_proximity",
                        candidate_target_object="excavator",
                        target_box_xyxy=[5, 1, 9, 8],
                    ),
                ],
            ),
            images=[
                QwenImageInput(
                    label="共同原图",
                    media_type="image/png",
                    data_url="data:image/png;base64,aW1hZ2U=",
                )
            ],
        )

        self.assertEqual(len(transport.calls), 1)
        self.assertIn(
            "多目标联合视觉事实提取器",
            transport.calls[0][2]["messages"][0]["content"],
        )
        provenance = result.as_dict()["provenance"]
        self.assertEqual(
            provenance["qwen_facts_prompt_version"],
            "construction-joint-visible-facts-v2",
        )
        self.assertEqual(
            provenance["qwen_enrichment_prompt_version"],
            "construction-joint-prompts-3-2-1-grounded-v6",
        )
        self.assertEqual(result.timings_ms["model_calls"], 1)

    def test_joint_generation_rejects_omitted_task_target(self):
        incomplete_facts = {
            **FACTS,
            "instance_count": 2,
            "task_targets": [
                {
                    "task_id": "tsk-1",
                    "target_object": "一名人员",
                    "instance_count": 1,
                    "visual_anchor": ["位于画面左侧"],
                },
                {
                    "task_id": "tsk-other",
                    "target_object": "未知目标",
                    "instance_count": 1,
                    "visual_anchor": ["位于画面右侧"],
                },
            ],
        }
        provider = Qwen25VLProvider(
            QwenProviderConfig(base_url="http://qwen25vl:8000/v1"),
            transport=FakeTransport(facts=incomplete_facts),
        )
        context = QwenJointVisualContext(
            asset_id="asset-1",
            targets=[
                QwenJointTarget(
                    task_id="tsk-1",
                    task_version=1,
                    category="unsafe",
                    candidate_target_object="person",
                    target_box_xyxy=[1, 1, 4, 8],
                ),
                QwenJointTarget(
                    task_id="tsk-2",
                    task_version=1,
                    category="unsafe",
                    candidate_target_object="safety vest",
                    target_box_xyxy=[2, 3, 4, 7],
                ),
            ],
        )

        with self.assertRaises(QwenContractError):
            provider.generate_joint(
                context=context,
                images=[
                    QwenImageInput(
                        label="共同原图",
                        media_type="image/png",
                        data_url="data:image/png;base64,aW1hZ2U=",
                    )
                ],
            )


if __name__ == "__main__":
    unittest.main()
