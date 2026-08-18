import base64
import json
import unittest

from annotation_service.pipeline.qwen.contract import (
    DEFAULT_QWEN_PROMPT_CONTEXT_PLACEHOLDER,
    DEFAULT_QWEN_PROMPT_TEMPLATE,
    QwenContractError,
    QwenImageInput,
    QwenJointVisualFacts,
    QwenPromptSet,
    QwenJointTarget,
    QwenJointVisualContext,
    QwenVisualContext,
    QwenVisualFacts,
    build_direct_prompt_messages,
    build_joint_prompt_enrichment_messages,
    build_joint_prompt_review_messages,
    build_joint_visual_facts_messages,
    build_prompt_enrichment_messages,
    build_visual_facts_messages,
    ground_prompt_set,
    ground_joint_prompt_set,
    parse_joint_prompt_set,
    parse_joint_visual_facts,
    parse_prompt_set,
    parse_visual_facts,
)


def facts_payload() -> dict:
    return {
        "target_object": "画面中央偏左的一名作业人员",
        "instance_count": 1,
        "visual_anchor": ["位于画面中央偏左", "穿深色上衣"],
        "mask_granularity": "人员整体",
        "visible_facts": ["人员头部没有可见安全帽"],
        "risk_semantics": "头部防护缺失",
    }


def prompt_payload() -> dict:
    return {
        "prompts": [
            {"prompt_id": "v1", "type": "visual", "text": "分割中央偏左人员。"},
            {"prompt_id": "v2", "type": "visual", "text": "标出深色上衣人员。"},
            {"prompt_id": "v3", "type": "visual", "text": "提取目标作业人员。"},
            {"prompt_id": "r1", "type": "risk", "text": "分割未戴安全帽人员。"},
            {"prompt_id": "r2", "type": "risk", "text": "标出头部防护缺失人员。"},
            {"prompt_id": "a1", "type": "agent", "text": "找出并分割违规人员。"},
        ]
    }


def joint_facts_payload() -> dict:
    return {
        **facts_payload(),
        "target_object": "一名人员及其旁边的挖掘机",
        "instance_count": 2,
        "task_targets": [
            {
                "task_id": "tsk-1",
                "target_object": "一名人员",
                "instance_count": 1,
                "visual_anchor": ["位于画面左侧"],
            },
            {
                "task_id": "tsk-2",
                "target_object": "一台挖掘机",
                "instance_count": 1,
                "visual_anchor": ["位于人员右侧"],
            },
        ],
    }


class QwenContractTest(unittest.TestCase):
    def test_visual_facts_parse_plain_or_fenced_json(self):
        plain = parse_visual_facts(
            json.dumps(facts_payload(), ensure_ascii=False)
        )
        fenced = parse_visual_facts(
            "```json\n"
            + json.dumps(facts_payload(), ensure_ascii=False)
            + "\n```"
        )
        self.assertEqual(plain.target_object, fenced.target_object)
        self.assertEqual(plain.instance_count, 1)

    def test_empty_optional_risk_semantics_is_normalized_to_none(self):
        payload = {**facts_payload(), "risk_semantics": "  "}

        facts = parse_visual_facts(
            json.dumps(payload, ensure_ascii=False)
        )
        prompts = ground_prompt_set(facts=facts)

        self.assertIsNone(facts.risk_semantics)
        self.assertIn(
            facts.visible_facts[0],
            {item.prompt_id: item.text for item in prompts.prompts}["risk-1"],
        )

    def test_visual_facts_reject_extra_or_duplicate_content(self):
        extra = {**facts_payload(), "unknown": "not allowed"}
        with self.assertRaises(QwenContractError):
            parse_visual_facts(json.dumps(extra, ensure_ascii=False))
        duplicate = facts_payload()
        duplicate["visible_facts"] = ["同一事实", "同一事实"]
        with self.assertRaises(QwenContractError):
            parse_visual_facts(json.dumps(duplicate, ensure_ascii=False))

    def test_prompt_set_allows_user_defined_count_and_types(self):
        parsed = parse_prompt_set(
            json.dumps(prompt_payload(), ensure_ascii=False)
        )
        self.assertEqual(len(parsed.prompts), 6)

        custom = prompt_payload()
        custom["prompts"] = custom["prompts"][:1]
        parsed_custom = parse_prompt_set(
            json.dumps(custom, ensure_ascii=False)
        )
        self.assertEqual(len(parsed_custom.prompts), 1)

        duplicate = prompt_payload()
        duplicate["prompts"][1]["text"] = duplicate["prompts"][0]["text"]
        deduplicated = parse_prompt_set(
            json.dumps(duplicate, ensure_ascii=False)
        )
        self.assertEqual(len(deduplicated.prompts), 5)

    def test_prompt_set_normalizes_plain_text_lines(self):
        parsed = parse_prompt_set(
            "1. 分割画面中央的人员。\n- 标出右侧的施工设备。"
        )

        self.assertEqual(
            [item.text for item in parsed.prompts],
            ["分割画面中央的人员。", "标出右侧的施工设备。"],
        )
        self.assertEqual(
            [item.prompt_id for item in parsed.prompts],
            ["prompt-1", "prompt-2"],
        )
        self.assertTrue(
            all(item.type.value == "visual" for item in parsed.prompts)
        )

    def test_prompt_set_normalizes_common_json_shapes(self):
        payloads = (
            ({"prompt": "分割目标人员。"}, ["分割目标人员。"]),
            (
                {"prompts": ["分割人员。", "分割设备。"]},
                ["分割人员。", "分割设备。"],
            ),
            (
                [
                    {"text": "分割人员。"},
                    {
                        "id": "custom-id",
                        "type": "unknown",
                        "content": "分割设备。",
                    },
                ],
                ["分割人员。", "分割设备。"],
            ),
        )
        for payload, expected in payloads:
            with self.subTest(payload=payload):
                parsed = parse_prompt_set(
                    json.dumps(payload, ensure_ascii=False)
                )
                self.assertEqual(
                    [item.text for item in parsed.prompts],
                    expected,
                )

    def test_prompt_set_repairs_ids_truncates_and_limits_items(self):
        payload = {
            "prompts": [
                {
                    "prompt_id": "same",
                    "type": "visual",
                    "text": f"提示词{index}" + "字" * 1100,
                }
                for index in range(55)
            ]
        }

        parsed = parse_prompt_set(json.dumps(payload, ensure_ascii=False))

        self.assertEqual(len(parsed.prompts), 50)
        self.assertEqual(len({item.prompt_id for item in parsed.prompts}), 50)
        self.assertTrue(all(len(item.text) <= 1000 for item in parsed.prompts))

    def test_prompt_set_rejects_empty_or_non_text_response(self):
        for raw in ("", "[]", '{"result": {"count": 0}}'):
            with self.subTest(raw=raw):
                with self.assertRaises(QwenContractError):
                    parse_prompt_set(raw)

    def test_single_target_prompts_are_deterministically_grounded(self):
        facts = QwenVisualFacts(**facts_payload())

        first = ground_prompt_set(facts=facts)
        second = ground_prompt_set(facts=facts)

        self.assertEqual(first, second)
        self.assertEqual(
            [prompt.type.value for prompt in first.prompts],
            ["visual", "visual", "visual", "risk", "risk", "agent"],
        )
        self.assertEqual(len({item.text for item in first.prompts}), 6)
        by_id = {item.prompt_id: item.text for item in first.prompts}
        self.assertIn(facts.target_object, by_id["visual-1"])
        self.assertIn(facts.visual_anchor[0], by_id["visual-2"])
        self.assertIn(facts.visible_facts[0], by_id["visual-3"])
        self.assertIn(facts.risk_semantics, by_id["risk-1"])
        self.assertIn(facts.mask_granularity, by_id["agent-1"])

    def test_joint_contract_requires_exact_task_coverage(self):
        facts = parse_joint_visual_facts(
            json.dumps(joint_facts_payload(), ensure_ascii=False),
            expected_task_ids=["tsk-1", "tsk-2"],
        )
        envelope = {
            **prompt_payload(),
            "covered_task_ids": ["tsk-1", "tsk-2"],
            "fact_consistent": True,
        }
        prompts = parse_joint_prompt_set(
            json.dumps(envelope, ensure_ascii=False),
            expected_task_ids=["tsk-1", "tsk-2"],
        )
        self.assertEqual(len(facts.task_targets), 2)
        self.assertEqual(len(prompts.prompts), 6)

        envelope["covered_task_ids"] = ["tsk-2", "tsk-1"]
        with self.assertRaises(QwenContractError):
            parse_joint_prompt_set(
                json.dumps(envelope, ensure_ascii=False),
                expected_task_ids=["tsk-1", "tsk-2"],
            )

    def test_joint_risk_and_agent_prompts_are_fact_grounded(self):
        facts = QwenJointVisualFacts(**joint_facts_payload())
        grounded = ground_joint_prompt_set(
            facts=facts,
        )
        by_id = {
            prompt.prompt_id: prompt.text
            for prompt in grounded.prompts
        }
        self.assertIn(facts.visible_facts[0], by_id["risk-1"])
        self.assertIn(facts.visual_anchor[0], by_id["risk-2"])
        self.assertIn(facts.target_object, by_id["agent-1"])
        self.assertIn(facts.target_object, by_id["visual-1"])
        self.assertNotIn("降低风险", by_id["risk-1"])

    def test_messages_separate_candidate_context_from_visible_facts(self):
        context = QwenVisualContext(
            asset_id="ast-1",
            category="helmet_missing",
            target_box_xyxy=[1, 2, 10, 20],
            hazard_evidence=["helmet was not matched in head region"],
        )
        facts_messages = build_visual_facts_messages(context)
        self.assertIn("不能当成事实", facts_messages[0]["content"])
        self.assertIn("helmet_missing", facts_messages[1]["content"])

        enrichment_messages = build_prompt_enrichment_messages(
            category=context.category,
            facts=QwenVisualFacts(**facts_payload()),
        )
        self.assertIn("同一目标", enrichment_messages[0]["content"])
        self.assertIn("3条visual", enrichment_messages[1]["content"])

    def test_visual_messages_include_ordered_image_inputs(self):
        context = QwenVisualContext(
            asset_id="asset-1",
            category="helmet_missing",
            target_box_xyxy=[1, 2, 3, 4],
        )
        encoded = base64.b64encode(b"image").decode("ascii")
        messages = build_visual_facts_messages(
            context,
            images=[
                QwenImageInput(
                    label="原图",
                    media_type="image/png",
                    data_url=f"data:image/png;base64,{encoded}",
                )
            ],
        )
        content = messages[1]["content"]
        self.assertIsInstance(content, list)
        self.assertEqual(content[-1]["type"], "image_url")
        self.assertTrue(
            content[-1]["image_url"]["url"].startswith(
                "data:image/png;base64,"
            )
        )

    def test_direct_custom_prompt_is_sent_verbatim_with_images_only(self):
        custom_prompt = "  只返回一条中文分割提示词。\n不要添加其他要求。  "
        image = QwenImageInput(
            label="该标签不得进入模型文本",
            media_type="image/png",
            data_url="data:image/png;base64,aW1hZ2U=",
        )

        messages = build_direct_prompt_messages(
            QwenVisualContext(
                asset_id="asset-1",
                category="helmet_missing",
                target_box_xyxy=[1, 2, 3, 4],
            ),
            custom_instruction=custom_prompt,
            images=[image],
        )

        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(
            messages[0]["content"],
            [
                {"type": "text", "text": custom_prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": image.data_url},
                },
            ],
        )

    def test_direct_default_prompt_contains_current_full_template(self):
        messages = build_direct_prompt_messages(
            QwenVisualContext(
                asset_id="asset-1",
                category="helmet_missing",
                target_box_xyxy=[1, 2, 3, 4],
            ),
            custom_instruction=None,
            images=[],
        )

        prompt_text = messages[0]["content"][0]["text"]
        self.assertNotIn(
            DEFAULT_QWEN_PROMPT_CONTEXT_PLACEHOLDER,
            prompt_text,
        )
        self.assertIn(
            DEFAULT_QWEN_PROMPT_CONTEXT_PLACEHOLDER,
            DEFAULT_QWEN_PROMPT_TEMPLATE,
        )
        self.assertIn("生成最终的图像分割 Prompt", prompt_text)
        self.assertIn("helmet_missing", prompt_text)
        self.assertIn('"prompt_id": "prompt-1"', prompt_text)
        self.assertIn("所有 text 必须使用简体中文", prompt_text)

    def test_joint_messages_require_all_targets_and_relationship(self):
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
                    task_version=2,
                    category="equipment_proximity",
                    candidate_target_object="excavator",
                    target_box_xyxy=[5, 1, 9, 8],
                ),
            ],
        )

        facts_messages = build_joint_visual_facts_messages(context)
        prompt_messages = build_joint_prompt_enrichment_messages(
            categories=[
                item.category for item in context.targets
            ],
            facts=QwenJointVisualFacts(**joint_facts_payload()),
        )
        review_messages = build_joint_prompt_review_messages(
            facts=QwenJointVisualFacts(**joint_facts_payload()),
            candidate_prompts=QwenPromptSet(**prompt_payload()),
        )

        self.assertIn("全部所选目标", facts_messages[0]["content"])
        self.assertIn("所有成员mask", prompt_messages[0]["content"])
        self.assertIn("fact_consistent", prompt_messages[1]["content"])
        self.assertIn("两个Task不", review_messages[0]["content"])
        self.assertIn(
            "equipment_proximity",
            prompt_messages[1]["content"],
        )


if __name__ == "__main__":
    unittest.main()
