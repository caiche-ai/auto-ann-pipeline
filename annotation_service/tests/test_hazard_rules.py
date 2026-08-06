import unittest

from annotation_service.hazard_rules import HazardRuleEngine


def detection(
    detection_id: str,
    entity: str,
    box: list[float],
    score: float = 0.9,
) -> dict:
    return {
        "detection_id": detection_id,
        "entity": entity,
        "box_xyxy": box,
        "box_score": score,
        "phrase_score": score,
    }


class HazardRuleEngineTest(unittest.TestCase):
    def setUp(self):
        self.engine = HazardRuleEngine()

    def infer(self, detections, categories):
        return self.engine.infer(
            detections=detections,
            requested_categories=categories,
            width=100,
            height=100,
        )

    def test_missing_helmet_is_candidate_not_final_truth(self):
        results = self.infer(
            [detection("det_person", "person", [10, 10, 50, 90])],
            ["helmet_missing"],
        )

        self.assertEqual(len(results), 1)
        candidate = results[0]
        self.assertEqual(candidate.category, "helmet_missing")
        self.assertEqual(candidate.target_detection_ids, ("det_person",))
        self.assertTrue(
            candidate.metadata["requires_visual_verification"]
        )
        self.assertTrue(candidate.metadata["negative_evidence"])
        self.assertLess(candidate.confidence, 0.9)

    def test_helmet_in_head_region_suppresses_missing_candidate(self):
        results = self.infer(
            [
                detection("det_person", "worker", [10, 10, 50, 90]),
                detection("det_helmet", "hard hat", [20, 8, 38, 28]),
            ],
            ["helmet_missing"],
        )
        self.assertEqual(results, [])

    def test_vest_in_torso_region_suppresses_no_jacket(self):
        results = self.infer(
            [
                detection("det_person", "person", [10, 10, 50, 90]),
                detection(
                    "det_vest",
                    "reflective vest",
                    [17, 32, 43, 64],
                ),
            ],
            ["no_jacket"],
        )
        self.assertEqual(results, [])

    def test_person_equipment_distance_creates_pair_evidence(self):
        near = self.infer(
            [
                detection("det_person", "person", [10, 10, 30, 80]),
                detection("det_excavator", "excavator", [35, 20, 90, 90]),
            ],
            ["equipment_proximity"],
        )
        far = self.infer(
            [
                detection("det_person", "person", [1, 1, 10, 20]),
                detection("det_excavator", "excavator", [80, 70, 99, 99]),
            ],
            ["equipment_proximity"],
        )

        self.assertEqual(len(near), 1)
        self.assertEqual(
            near[0].target_detection_ids,
            ("det_person", "det_excavator"),
        )
        self.assertEqual(
            near[0].metadata["equipment_entity"],
            "excavator",
        )
        self.assertEqual(far, [])

    def test_opening_with_nearby_guardrail_is_not_flagged(self):
        protected = self.infer(
            [
                detection("det_opening", "floor hole", [30, 40, 60, 70]),
                detection("det_guardrail", "guardrail", [24, 35, 66, 45]),
            ],
            ["opening_unprotected"],
        )
        unprotected = self.infer(
            [detection("det_opening", "opening", [30, 40, 60, 70])],
            ["opening_unprotected"],
        )

        self.assertEqual(protected, [])
        self.assertEqual(len(unprotected), 1)
        self.assertEqual(
            unprotected[0].target_entity,
            "opening",
        )

    def test_safe_is_never_inferred_from_box_presence_alone(self):
        results = self.infer(
            [
                detection("det_person", "person", [10, 10, 50, 90]),
                detection("det_helmet", "helmet", [20, 8, 38, 28]),
            ],
            ["safe"],
        )
        self.assertEqual(results, [])

    def test_unsafe_wraps_a_concrete_rule_with_provenance(self):
        results = self.infer(
            [
                detection("det_person", "person", [10, 10, 30, 80]),
                detection("det_excavator", "excavator", [35, 20, 90, 90]),
            ],
            ["unsafe"],
        )

        self.assertGreaterEqual(len(results), 1)
        self.assertTrue(all(item.category == "unsafe" for item in results))
        self.assertTrue(
            all("derived_category" in item.metadata for item in results)
        )
        self.assertTrue(
            all(item.rule_id.startswith("unsafe.from.") for item in results)
        )


if __name__ == "__main__":
    unittest.main()
