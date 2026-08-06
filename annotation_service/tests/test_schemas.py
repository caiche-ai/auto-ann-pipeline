import unittest

from pydantic import ValidationError

from annotation_service.schemas import (
    AnnotationContent,
    AnnotationPrompt,
    CreateJobRequest,
    CreateMaskCandidateRequest,
    CreateReleaseRequest,
    PolygonShape,
    ReleaseTaskFilter,
    SplitPolicy,
)


class SchemaTest(unittest.TestCase):
    def test_job_assets_must_be_unique_and_prompt_is_free_form(self):
        with self.assertRaises(ValidationError):
            CreateJobRequest(
                asset_ids=["a", "a"],
                grounding_prompt="person",
                pipeline_version="v1",
            )

        request = CreateJobRequest(
            asset_ids=["a"],
            grounding_prompt=" 任意中文 Prompt：设备旁的人员！ ",
        )
        self.assertEqual(
            request.grounding_prompt,
            "任意中文 Prompt：设备旁的人员！",
        )
        self.assertEqual(
            request.grounding_prompt_normalization_mode,
            "terminal_period",
        )
        self.assertEqual(
            request.grounding_prompt_normalization_profile,
            "construction_safety_v1",
        )
        self.assertEqual(
            request.grounding_prompt_translation_failure_policy,
            "fallback_canonical_terms",
        )

        custom = CreateJobRequest(
            asset_ids=["a"],
            grounding_prompt="person near excavator",
            grounding_prompt_normalization_mode="off",
            grounding_prompt_normalization_profile="construction_safety_v1",
        )
        self.assertEqual(
            custom.grounding_prompt_normalization_mode,
            "off",
        )

        open_semantic = CreateJobRequest(
            asset_ids=["a"],
            grounding_prompt="找出蓝色设备旁的施工人员",
            grounding_prompt_normalization_mode="llm_grounding_caption",
            grounding_prompt_normalization_profile=(
                "open_semantic_zh_en_v1"
            ),
            grounding_prompt_translation_failure_policy="fail_job",
        )
        self.assertEqual(
            open_semantic.grounding_prompt_normalization_mode,
            "llm_grounding_caption",
        )
        with self.assertRaises(ValidationError):
            CreateJobRequest(
                asset_ids=["a"],
                grounding_prompt="施工人员",
                grounding_prompt_normalization_mode=(
                    "llm_grounding_caption"
                ),
                grounding_prompt_normalization_profile=(
                    "construction_safety_v1"
                ),
            )

        with self.assertRaises(ValidationError):
            CreateJobRequest(
                asset_ids=["a"],
                grounding_prompt="   ",
                pipeline_version="v1",
            )

    def test_boxes_require_positive_area(self):
        with self.assertRaises(ValidationError):
            CreateMaskCandidateRequest(
                expected_version=1,
                box_xyxy=[10, 10, 10, 20],
            )

    def test_polygon_requires_three_points(self):
        with self.assertRaises(ValidationError):
            PolygonShape(
                shape_id="shape-1",
                label="target",
                points=[[0, 0], [1, 1]],
            )

    def test_prompt_is_trimmed(self):
        prompt = AnnotationPrompt(
            prompt_id="p1",
            type="visual",
            text="  标出目标。  ",
        )
        self.assertEqual(prompt.text, "标出目标。")

    def test_draft_can_be_incomplete_but_has_structured_fields(self):
        draft = AnnotationContent(
            target_object="",
            instance_count=1,
            visual_anchor=[],
            mask_granularity="",
            shapes=[],
            prompts=[],
        )
        self.assertEqual(draft.prompts, [])

    def test_split_ratios_must_sum_to_one(self):
        with self.assertRaises(ValidationError):
            SplitPolicy(
                train_ratio=0.8,
                val_ratio=0.2,
                golden_ratio=0.1,
                seed=42,
            )

    def test_release_name_is_safe(self):
        valid = CreateReleaseRequest(
            name="ReasonSegGroundedV1",
            task_filter=ReleaseTaskFilter(),
            split_policy=SplitPolicy(
                train_ratio=0.8,
                val_ratio=0.1,
                golden_ratio=0.1,
                seed=42,
            ),
        )
        self.assertEqual(valid.name, "ReasonSegGroundedV1")

        with self.assertRaises(ValidationError):
            CreateReleaseRequest(
                name="../unsafe",
                task_filter=ReleaseTaskFilter(),
                split_policy=SplitPolicy(
                    train_ratio=0.8,
                    val_ratio=0.1,
                    golden_ratio=0.1,
                    seed=42,
                ),
            )
        with self.assertRaises(ValidationError):
            ReleaseTaskFilter(categories=[])
        with self.assertRaises(ValidationError):
            ReleaseTaskFilter(
                categories=["helmet_missing", "helmet_missing"]
            )

    def test_unknown_fields_are_rejected(self):
        with self.assertRaises(ValidationError):
            CreateJobRequest(
                asset_ids=["a"],
                grounding_prompt="person",
                pipeline_version="v1",
                unexpected=True,
            )


if __name__ == "__main__":
    unittest.main()
