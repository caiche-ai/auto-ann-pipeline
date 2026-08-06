import base64
import json
import unittest

from annotation_service.qwen_contract import (
    QwenContractError,
    QwenImageInput,
    QwenJointVisualFacts,
    QwenPromptSet,
    QwenJointTarget,
    QwenJointVisualContext,
    QwenVisualContext,
    QwenVisualFacts,
    build_joint_prompt_enrichment_messages,
    build_joint_prompt_review_messages,
    build_joint_visual_facts_messages,
    build_prompt_enrichment_messages,
    build_visual_facts_messages,
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

    def test_visual_facts_reject_extra_or_duplicate_content(self):
        extra = {**facts_payload(), "unknown": "not allowed"}
        with self.assertRaises(QwenContractError):
            parse_visual_facts(json.dumps(extra, ensure_ascii=False))
        duplicate = facts_payload()
        duplicate["visible_facts"] = ["同一事实", "同一事实"]
        with self.assertRaises(QwenContractError):
            parse_visual_facts(json.dumps(duplicate, ensure_ascii=False))

    def test_prompt_set_requires_exact_three_two_one(self):
        parsed = parse_prompt_set(
            json.dumps(prompt_payload(), ensure_ascii=False)
        )
        self.assertEqual(len(parsed.prompts), 6)

        invalid = prompt_payload()
        invalid["prompts"][5]["type"] = "visual"
        with self.assertRaises(QwenContractError):
            parse_prompt_set(json.dumps(invalid, ensure_ascii=False))

        duplicate = prompt_payload()
        duplicate["prompts"][1]["text"] = duplicate["prompts"][0]["text"]
        with self.assertRaises(QwenContractError):
            parse_prompt_set(json.dumps(duplicate, ensure_ascii=False))

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
