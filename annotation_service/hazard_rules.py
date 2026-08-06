from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from .schemas import AnnotationCategory


Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class HazardRuleConfig:
    version: str = "construction-hazard-rules-v1"
    protective_item_ioa_threshold: float = 0.30
    equipment_proximity_diagonal_ratio: float = 0.08
    opening_protection_expansion_ratio: float = 0.20
    minimum_detection_score: float = 0.25


@dataclass(frozen=True)
class HazardCandidateResult:
    category: str
    target_entity: str
    target_detection_ids: tuple[str, ...]
    box_xyxy: Box
    confidence: float
    rule_id: str
    rule_version: str
    evidence: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_storage_payload(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "target_entity": self.target_entity,
            "target_detection_ids": list(self.target_detection_ids),
            "box_xyxy": list(self.box_xyxy),
            "confidence": self.confidence,
            "rule_id": self.rule_id,
            "rule_version": self.rule_version,
            "evidence": list(self.evidence),
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class _Detection:
    detection_id: str
    entity: str
    box: Box
    score: float


_ALIASES = {
    "people": "person",
    "worker": "person",
    "construction worker": "person",
    "hard hat": "helmet",
    "hardhat": "helmet",
    "safety helmet": "helmet",
    "reflective jacket": "safety vest",
    "reflective vest": "safety vest",
    "high visibility vest": "safety vest",
    "hi vis vest": "safety vest",
    "body harness": "safety harness",
    "fall arrest harness": "safety harness",
    "floor opening": "opening",
    "floor hole": "opening",
    "hole": "opening",
    "edge": "platform edge",
    "safety railing": "guardrail",
    "railing": "guardrail",
    "barrier": "barricade",
    "construction debris": "debris",
    "rubble": "debris",
}

_RULE_ORDER = (
    AnnotationCategory.HELMET_MISSING,
    AnnotationCategory.NO_HELMET,
    AnnotationCategory.NO_JACKET,
    AnnotationCategory.HARNESS_MISSING,
    AnnotationCategory.EQUIPMENT_PROXIMITY,
    AnnotationCategory.OPENING_UNPROTECTED,
    AnnotationCategory.GUARDRAIL_MISSING,
    AnnotationCategory.POOR_HOUSEKEEPING,
)

_GENERIC_UNSAFE_RULE_ORDER = (
    AnnotationCategory.EQUIPMENT_PROXIMITY,
    AnnotationCategory.OPENING_UNPROTECTED,
    AnnotationCategory.GUARDRAIL_MISSING,
    AnnotationCategory.POOR_HOUSEKEEPING,
)


def _normalize_entity(value: str) -> str:
    normalized = re.sub(r"\s+", " ", value.lower()).strip(" ._-")
    return _ALIASES.get(normalized, normalized)


def _area(box: Box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _intersection(first: Box, second: Box) -> float:
    width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    return width * height


def _ioa(candidate: Box, region: Box) -> float:
    candidate_area = _area(candidate)
    return _intersection(candidate, region) / candidate_area if candidate_area else 0.0


def _center(box: Box) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def _contains(region: Box, point: tuple[float, float]) -> bool:
    return (
        region[0] <= point[0] <= region[2]
        and region[1] <= point[1] <= region[3]
    )


def _person_region(person: Box, top_ratio: float, bottom_ratio: float) -> Box:
    width = person[2] - person[0]
    height = person[3] - person[1]
    return (
        max(0.0, person[0] - width * 0.08),
        max(0.0, person[1] + height * top_ratio),
        person[2] + width * 0.08,
        person[1] + height * bottom_ratio,
    )


def _expanded(box: Box, ratio: float, width: int, height: int) -> Box:
    delta_x = (box[2] - box[0]) * ratio
    delta_y = (box[3] - box[1]) * ratio
    return (
        max(0.0, box[0] - delta_x),
        max(0.0, box[1] - delta_y),
        min(float(width), box[2] + delta_x),
        min(float(height), box[3] + delta_y),
    )


def _edge_gap(first: Box, second: Box) -> float:
    horizontal = max(first[0] - second[2], second[0] - first[2], 0.0)
    vertical = max(first[1] - second[3], second[1] - first[3], 0.0)
    return math.hypot(horizontal, vertical)


def _union(first: Box, second: Box) -> Box:
    return (
        min(first[0], second[0]),
        min(first[1], second[1]),
        max(first[2], second[2]),
        max(first[3], second[3]),
    )


def _round_confidence(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 6)


class HazardRuleEngine:
    """Derive reviewable hazard candidates from GroundingDINO boxes.

    The engine intentionally produces candidates, not final labels. In
    particular, missing-PPE rules use absence of a matched protective-item
    box as weak negative evidence and mark the result for visual verification.
    """

    def __init__(self, config: HazardRuleConfig | None = None):
        self.config = config or HazardRuleConfig()

    def infer(
        self,
        *,
        detections: Sequence[dict[str, Any]],
        requested_categories: Sequence[str | AnnotationCategory],
        width: int,
        height: int,
    ) -> list[HazardCandidateResult]:
        if width < 1 or height < 1:
            raise ValueError("image dimensions must be positive")
        normalized = self._normalize_detections(detections)
        requested = tuple(
            value
            if isinstance(value, AnnotationCategory)
            else AnnotationCategory(str(value))
            for value in requested_categories
        )
        wants_unsafe = AnnotationCategory.UNSAFE in requested
        explicit_categories = tuple(
            category for category in _RULE_ORDER if category in requested
        )
        categories = explicit_categories
        if wants_unsafe:
            categories = tuple(
                dict.fromkeys(
                    (*explicit_categories, *_GENERIC_UNSAFE_RULE_ORDER)
                )
            )

        specific: list[HazardCandidateResult] = []
        for category in categories:
            if category in {
                AnnotationCategory.HELMET_MISSING,
                AnnotationCategory.NO_HELMET,
            }:
                specific.extend(
                    self._missing_ppe(
                        normalized,
                        category=category,
                        item_entities={"helmet"},
                        region_ratios=(0.0, 0.38),
                        rule_id="ppe.helmet_absent_in_head_region",
                        target_label="helmet",
                    )
                )
            elif category == AnnotationCategory.NO_JACKET:
                specific.extend(
                    self._missing_ppe(
                        normalized,
                        category=category,
                        item_entities={"safety vest"},
                        region_ratios=(0.20, 0.78),
                        rule_id="ppe.vest_absent_in_torso_region",
                        target_label="safety vest",
                    )
                )
            elif category == AnnotationCategory.HARNESS_MISSING:
                specific.extend(
                    self._missing_ppe(
                        normalized,
                        category=category,
                        item_entities={"safety harness"},
                        region_ratios=(0.18, 0.82),
                        rule_id="ppe.harness_absent_in_torso_region",
                        target_label="safety harness",
                        confidence_factor=0.55,
                    )
                )
            elif category == AnnotationCategory.EQUIPMENT_PROXIMITY:
                specific.extend(
                    self._equipment_proximity(normalized, width, height)
                )
            elif category == AnnotationCategory.OPENING_UNPROTECTED:
                specific.extend(
                    self._opening_unprotected(normalized, width, height)
                )
            elif category == AnnotationCategory.GUARDRAIL_MISSING:
                specific.extend(
                    self._guardrail_missing(normalized, width, height)
                )
            elif category == AnnotationCategory.POOR_HOUSEKEEPING:
                specific.extend(self._poor_housekeeping(normalized))

        results = [
            item
            for item in specific
            if AnnotationCategory(item.category) in explicit_categories
        ]
        if wants_unsafe:
            results.extend(
                self._as_unsafe(item)
                for item in specific
                if AnnotationCategory(item.category)
                in _GENERIC_UNSAFE_RULE_ORDER
            )
        return self._deduplicate(results)

    def _normalize_detections(
        self,
        detections: Sequence[dict[str, Any]],
    ) -> list[_Detection]:
        normalized: list[_Detection] = []
        for item in detections:
            detection_id = str(item.get("detection_id", "")).strip()
            entity = _normalize_entity(str(item.get("entity", "")))
            box_value = item.get("box_xyxy")
            if not detection_id or not entity:
                raise ValueError("detections require detection_id and entity")
            if not isinstance(box_value, (list, tuple)) or len(box_value) != 4:
                raise ValueError("detection box_xyxy must contain four coordinates")
            box = tuple(float(value) for value in box_value)
            if min(box) < 0 or box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError("detection box must have non-negative positive area")
            box_score = float(item.get("box_score", 0.0))
            phrase_score = float(item.get("phrase_score", box_score))
            score = min(box_score, phrase_score)
            if score >= self.config.minimum_detection_score:
                normalized.append(
                    _Detection(
                        detection_id=detection_id,
                        entity=entity,
                        box=box,
                        score=score,
                    )
                )
        return sorted(normalized, key=lambda item: item.detection_id)

    @staticmethod
    def _by_entity(
        detections: Iterable[_Detection],
        entities: set[str],
    ) -> list[_Detection]:
        return [item for item in detections if item.entity in entities]

    def _matched_to_region(
        self,
        item: _Detection,
        region: Box,
    ) -> bool:
        return (
            _contains(region, _center(item.box))
            or _ioa(item.box, region)
            >= self.config.protective_item_ioa_threshold
        )

    def _missing_ppe(
        self,
        detections: Sequence[_Detection],
        *,
        category: AnnotationCategory,
        item_entities: set[str],
        region_ratios: tuple[float, float],
        rule_id: str,
        target_label: str,
        confidence_factor: float = 0.65,
    ) -> list[HazardCandidateResult]:
        people = self._by_entity(detections, {"person"})
        items = self._by_entity(detections, item_entities)
        results: list[HazardCandidateResult] = []
        for person in people:
            region = _person_region(person.box, *region_ratios)
            matches = [
                item for item in items if self._matched_to_region(item, region)
            ]
            if matches:
                continue
            results.append(
                HazardCandidateResult(
                    category=category.value,
                    target_entity="person",
                    target_detection_ids=(person.detection_id,),
                    box_xyxy=person.box,
                    confidence=_round_confidence(
                        person.score * confidence_factor
                    ),
                    rule_id=rule_id,
                    rule_version=self.config.version,
                    evidence=(
                        f"detected person {person.detection_id}",
                        (
                            f"no {target_label} detection matched the "
                            f"person region"
                        ),
                    ),
                    metadata={
                        "requires_visual_verification": True,
                        "negative_evidence": True,
                        "matched_protective_detection_ids": [],
                    },
                )
            )
        return results

    def _equipment_proximity(
        self,
        detections: Sequence[_Detection],
        width: int,
        height: int,
    ) -> list[HazardCandidateResult]:
        people = self._by_entity(detections, {"person"})
        equipment = self._by_entity(
            detections,
            {"excavator", "construction vehicle", "crane", "forklift"},
        )
        diagonal = math.hypot(width, height)
        threshold = diagonal * self.config.equipment_proximity_diagonal_ratio
        results: list[HazardCandidateResult] = []
        for person in people:
            for machine in equipment:
                gap = _edge_gap(person.box, machine.box)
                if gap > threshold:
                    continue
                proximity = 1.0 - gap / threshold if threshold else 1.0
                results.append(
                    HazardCandidateResult(
                        category=AnnotationCategory.EQUIPMENT_PROXIMITY.value,
                        target_entity="person",
                        target_detection_ids=(
                            person.detection_id,
                            machine.detection_id,
                        ),
                        box_xyxy=person.box,
                        confidence=_round_confidence(
                            min(person.score, machine.score)
                            * (0.70 + 0.30 * proximity)
                        ),
                        rule_id="spatial.person_equipment_proximity",
                        rule_version=self.config.version,
                        evidence=(
                            f"detected person {person.detection_id}",
                            (
                                f"detected {machine.entity} "
                                f"{machine.detection_id}"
                            ),
                            f"box edge gap is {gap:.2f} pixels",
                        ),
                        metadata={
                            "requires_visual_verification": True,
                            "equipment_entity": machine.entity,
                            "edge_gap_pixels": round(gap, 4),
                            "threshold_pixels": round(threshold, 4),
                            "context_box_xyxy": list(
                                _union(person.box, machine.box)
                            ),
                        },
                    )
                )
        return results

    def _opening_unprotected(
        self,
        detections: Sequence[_Detection],
        width: int,
        height: int,
    ) -> list[HazardCandidateResult]:
        openings = self._by_entity(detections, {"opening"})
        protections = self._by_entity(
            detections,
            {"guardrail", "barricade", "cover"},
        )
        results: list[HazardCandidateResult] = []
        for opening in openings:
            region = _expanded(
                opening.box,
                self.config.opening_protection_expansion_ratio,
                width,
                height,
            )
            matches = [
                item
                for item in protections
                if _intersection(item.box, region) > 0
                or _contains(region, _center(item.box))
            ]
            if matches:
                continue
            results.append(
                HazardCandidateResult(
                    category=AnnotationCategory.OPENING_UNPROTECTED.value,
                    target_entity="opening",
                    target_detection_ids=(opening.detection_id,),
                    box_xyxy=opening.box,
                    confidence=_round_confidence(opening.score * 0.65),
                    rule_id="protection.opening_without_nearby_barrier",
                    rule_version=self.config.version,
                    evidence=(
                        f"detected opening {opening.detection_id}",
                        "no cover, guardrail, or barricade box matched nearby",
                    ),
                    metadata={
                        "requires_visual_verification": True,
                        "negative_evidence": True,
                        "matched_protection_detection_ids": [],
                    },
                )
            )
        return results

    def _guardrail_missing(
        self,
        detections: Sequence[_Detection],
        width: int,
        height: int,
    ) -> list[HazardCandidateResult]:
        edges = self._by_entity(
            detections,
            {"platform edge", "opening"},
        )
        guardrails = self._by_entity(detections, {"guardrail"})
        results: list[HazardCandidateResult] = []
        for edge in edges:
            region = _expanded(
                edge.box,
                self.config.opening_protection_expansion_ratio,
                width,
                height,
            )
            matches = [
                item
                for item in guardrails
                if _intersection(item.box, region) > 0
                or _contains(region, _center(item.box))
            ]
            if matches:
                continue
            results.append(
                HazardCandidateResult(
                    category=AnnotationCategory.GUARDRAIL_MISSING.value,
                    target_entity=edge.entity,
                    target_detection_ids=(edge.detection_id,),
                    box_xyxy=edge.box,
                    confidence=_round_confidence(edge.score * 0.60),
                    rule_id="protection.edge_without_nearby_guardrail",
                    rule_version=self.config.version,
                    evidence=(
                        f"detected {edge.entity} {edge.detection_id}",
                        "no guardrail detection matched the expanded edge region",
                    ),
                    metadata={
                        "requires_visual_verification": True,
                        "negative_evidence": True,
                        "matched_guardrail_detection_ids": [],
                    },
                )
            )
        return results

    def _poor_housekeeping(
        self,
        detections: Sequence[_Detection],
    ) -> list[HazardCandidateResult]:
        debris = self._by_entity(detections, {"debris"})
        materials = self._by_entity(detections, {"construction material"})
        walkways = self._by_entity(detections, {"walkway"})
        results = [
            HazardCandidateResult(
                category=AnnotationCategory.POOR_HOUSEKEEPING.value,
                target_entity="debris",
                target_detection_ids=(item.detection_id,),
                box_xyxy=item.box,
                confidence=_round_confidence(item.score * 0.80),
                rule_id="housekeeping.visible_debris",
                rule_version=self.config.version,
                evidence=(f"detected debris {item.detection_id}",),
                metadata={"requires_visual_verification": True},
            )
            for item in debris
        ]
        for material in materials:
            for walkway in walkways:
                overlap = _ioa(material.box, walkway.box)
                if overlap < self.config.protective_item_ioa_threshold:
                    continue
                results.append(
                    HazardCandidateResult(
                        category=AnnotationCategory.POOR_HOUSEKEEPING.value,
                        target_entity="construction material",
                        target_detection_ids=(
                            material.detection_id,
                            walkway.detection_id,
                        ),
                        box_xyxy=material.box,
                        confidence=_round_confidence(
                            min(material.score, walkway.score)
                            * (0.65 + min(overlap, 1.0) * 0.20)
                        ),
                        rule_id="housekeeping.material_on_walkway",
                        rule_version=self.config.version,
                        evidence=(
                            (
                                "construction material "
                                f"{material.detection_id} overlaps walkway "
                                f"{walkway.detection_id}"
                            ),
                            f"material intersection-over-area is {overlap:.3f}",
                        ),
                        metadata={
                            "requires_visual_verification": True,
                            "material_walkway_ioa": round(overlap, 6),
                        },
                    )
                )
        return results

    def _as_unsafe(
        self,
        candidate: HazardCandidateResult,
    ) -> HazardCandidateResult:
        return HazardCandidateResult(
            category=AnnotationCategory.UNSAFE.value,
            target_entity=candidate.target_entity,
            target_detection_ids=candidate.target_detection_ids,
            box_xyxy=candidate.box_xyxy,
            confidence=candidate.confidence,
            rule_id=f"unsafe.from.{candidate.rule_id}",
            rule_version=candidate.rule_version,
            evidence=(
                f"derived concrete hazard category: {candidate.category}",
                *candidate.evidence,
            ),
            metadata={
                **candidate.metadata,
                "derived_category": candidate.category,
            },
        )

    @staticmethod
    def _deduplicate(
        candidates: Sequence[HazardCandidateResult],
    ) -> list[HazardCandidateResult]:
        unique: dict[
            tuple[str, tuple[str, ...], str],
            HazardCandidateResult,
        ] = {}
        for candidate in candidates:
            key = (
                candidate.category,
                candidate.target_detection_ids,
                candidate.rule_id,
            )
            unique.setdefault(key, candidate)
        return sorted(
            unique.values(),
            key=lambda item: (
                item.category,
                item.target_detection_ids,
                item.rule_id,
            ),
        )
