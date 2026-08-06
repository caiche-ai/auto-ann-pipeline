import unittest

from annotation_service.errors import AnnotationValidationError
from annotation_service.validation import (
    normalize_request_id,
    parse_metadata_json,
    validate_annotation_for_submission,
    validate_group_id,
)


class ValidationTest(unittest.TestCase):
    def test_group_id(self):
        self.assertEqual(
            validate_group_id("site01:video03"),
            "site01:video03",
        )
        with self.assertRaises(ValueError):
            validate_group_id("../site")

    def test_metadata_must_be_an_object(self):
        self.assertEqual(
            parse_metadata_json('{"camera":"north"}', max_chars=100),
            {"camera": "north"},
        )
        with self.assertRaises(ValueError):
            parse_metadata_json("[1,2,3]", max_chars=100)

    def test_metadata_limit(self):
        with self.assertRaises(ValueError):
            parse_metadata_json('{"value":"long"}', max_chars=5)

    def test_request_id_is_trimmed_and_bounded(self):
        self.assertEqual(normalize_request_id("  abc  "), "abc")
        self.assertEqual(len(normalize_request_id("x" * 200)), 128)

    def test_submission_requires_mask_and_three_two_one_prompts(self):
        with self.assertRaises(AnnotationValidationError) as context:
            validate_annotation_for_submission(
                {
                    "target_object": "一名作业人员",
                    "mask_granularity": "人员整体",
                    "shapes": [],
                    "prompts": [],
                },
                width=100,
                height=80,
                category="helmet_missing",
            )
        fields = {item["field"] for item in context.exception.details}
        self.assertIn("annotation.shapes", fields)
        self.assertIn("annotation.prompts", fields)

    def test_submission_rejects_zero_area_and_out_of_bounds_polygon(self):
        annotation = {
            "target_object": "一名作业人员",
            "mask_granularity": "人员整体",
            "shapes": [
                {
                    "label": "target",
                    "points": [[0, 0], [5, 5], [10, 10], [120, 10]],
                }
            ],
            "prompts": [
                {"type": "visual", "text": "v1"},
                {"type": "visual", "text": "v2"},
                {"type": "visual", "text": "v3"},
                {"type": "risk", "text": "r1"},
                {"type": "risk", "text": "r2"},
                {"type": "agent", "text": "a1"},
            ],
        }
        with self.assertRaises(AnnotationValidationError) as context:
            validate_annotation_for_submission(
                annotation,
                width=100,
                height=80,
                category="helmet_missing",
            )
        reasons = [item["reason"] for item in context.exception.details]
        self.assertTrue(any("inside" in reason for reason in reasons))

    def test_complete_submission_is_accepted(self):
        validate_annotation_for_submission(
            {
                "target_object": "画面中央的一名作业人员",
                "mask_granularity": "人员整体",
                "shapes": [
                    {
                        "shape_id": "shape-1",
                        "label": "target",
                        "points": [[1, 1], [8, 1], [8, 8], [1, 8]],
                    }
                ],
                "prompts": [
                    {"prompt_id": "v1", "type": "visual", "text": "v1"},
                    {"prompt_id": "v2", "type": "visual", "text": "v2"},
                    {"prompt_id": "v3", "type": "visual", "text": "v3"},
                    {"prompt_id": "r1", "type": "risk", "text": "r1"},
                    {"prompt_id": "r2", "type": "risk", "text": "r2"},
                    {"prompt_id": "a1", "type": "agent", "text": "a1"},
                ],
            },
            width=10,
            height=10,
            category="helmet_missing",
        )


if __name__ == "__main__":
    unittest.main()
