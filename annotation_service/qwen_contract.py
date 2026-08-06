from __future__ import annotations

import json
from dataclasses import dataclass
from collections import Counter
from typing import Any, List, Optional

from pydantic import Field, root_validator, validator

from .schemas import (
    AnnotationCategory,
    AnnotationPrompt,
    Detection,
    PromptType,
    StrictModel,
)


QWEN_FACTS_PROMPT_VERSION = "construction-visible-facts-v1"
QWEN_ENRICHMENT_PROMPT_VERSION = "construction-prompts-3-2-1-v1"

RISK_SEMANTIC_BOUNDARIES = {
    AnnotationCategory.OPENING_UNPROTECTED: (
        "可描述未封闭、未加盖、洞口防护缺失或坠落风险；不得虚构人员正在坠落。"
    ),
    AnnotationCategory.GUARDRAIL_MISSING: (
        "可描述临边防护缺失、护栏未设置或坠落风险；必须明确具体边沿、开口或构件。"
    ),
    AnnotationCategory.HARNESS_MISSING: (
        "只有图中确有对应人员且mask标人时，才可描述未使用安全带或防坠措施不足。"
    ),
    AnnotationCategory.HELMET_MISSING: (
        "目标必须是mask中的人员；可描述未佩戴安全帽、头部防护缺失或头部伤害风险。"
    ),
    AnnotationCategory.NO_HELMET: (
        "目标必须是mask中的人员；可描述未佩戴安全帽、头部防护缺失或头部伤害风险。"
    ),
    AnnotationCategory.NO_JACKET: (
        "只可描述未穿反光背心或可视性防护不足，不得写成未穿普通外套。"
    ),
    AnnotationCategory.EQUIPMENT_PROXIMITY: (
        "只有人员、设备关系和mask共同支持时，才可描述人机距离过近、碰撞或挤压风险。"
    ),
    AnnotationCategory.POOR_HOUSEKEEPING: (
        "只有画面支持时，才可描述材料堆放混乱、通道受阻或绊倒风险。"
    ),
    AnnotationCategory.SAFE: (
        "只能描述图中和mask支持的合规对象或安全状态，不得加入不存在的隐患。"
    ),
    AnnotationCategory.UNSAFE: (
        "必须具体化不安全对象；不能只写危险区域、不安全目标或违规位置。"
    ),
}


class QwenContractError(ValueError):
    """Raised when a Qwen response does not satisfy the frozen contract."""


@dataclass(frozen=True)
class QwenImageInput:
    label: str
    media_type: str
    data_url: str

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError("image label must not be blank")
        if self.media_type not in {"image/jpeg", "image/png"}:
            raise ValueError("Qwen image must be JPEG or PNG")
        expected_prefix = f"data:{self.media_type};base64,"
        if not self.data_url.startswith(expected_prefix):
            raise ValueError(
                f"image data URL must start with {expected_prefix}"
            )


class QwenVisualContext(StrictModel):
    asset_id: str
    category: AnnotationCategory
    target_box_xyxy: List[float]
    target_detection_ids: List[str] = Field(default_factory=list)
    detections: List[Detection] = Field(default_factory=list)
    hazard_evidence: List[str] = Field(default_factory=list)
    requires_visual_verification: bool = True
    mask_available: bool = False

    @validator("target_box_xyxy")
    def target_box_is_valid(cls, value: List[float]) -> List[float]:
        if len(value) != 4:
            raise ValueError("target_box_xyxy must contain four coordinates")
        x1, y1, x2, y2 = value
        if min(value) < 0 or x2 <= x1 or y2 <= y1:
            raise ValueError(
                "target_box_xyxy must have non-negative positive area"
            )
        return value

    @validator("target_detection_ids", "hazard_evidence")
    def list_values_are_unique(cls, value: List[str]) -> List[str]:
        normalized = [item.strip() for item in value]
        if any(not item for item in normalized):
            raise ValueError("list values must not be blank")
        if len(normalized) != len(set(normalized)):
            raise ValueError("list values must be unique")
        return normalized


class QwenVisualFacts(StrictModel):
    target_object: str = Field(..., min_length=1, max_length=300)
    instance_count: int = Field(..., ge=1)
    visual_anchor: List[str] = Field(..., min_items=1, max_items=10)
    mask_granularity: str = Field(..., min_length=1, max_length=200)
    visible_facts: List[str] = Field(..., min_items=1, max_items=20)
    risk_semantics: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=500,
    )

    @validator(
        "target_object",
        "mask_granularity",
        "risk_semantics",
        pre=True,
    )
    def text_is_trimmed(cls, value):
        return value.strip() if isinstance(value, str) else value

    @validator("visual_anchor", "visible_facts")
    def facts_are_non_empty_and_unique(cls, value: List[str]) -> List[str]:
        normalized = [item.strip() for item in value]
        if any(not item for item in normalized):
            raise ValueError("facts must not be blank")
        if len(normalized) != len(set(normalized)):
            raise ValueError("facts must be unique")
        return normalized


class QwenPromptSet(StrictModel):
    prompts: List[AnnotationPrompt] = Field(
        ...,
        min_items=6,
        max_items=6,
    )

    @root_validator(skip_on_failure=True)
    def prompts_follow_three_two_one(cls, values):
        prompts = values.get("prompts") or []
        counts = Counter(item.type for item in prompts)
        expected = {
            PromptType.VISUAL: 3,
            PromptType.RISK: 2,
            PromptType.AGENT: 1,
        }
        for prompt_type, expected_count in expected.items():
            if counts.get(prompt_type, 0) != expected_count:
                raise ValueError(
                    f"expected {expected_count} {prompt_type.value} prompts"
                )
        texts = [item.text for item in prompts]
        if len(texts) != len(set(texts)):
            raise ValueError("prompt texts must be unique")
        prompt_ids = [item.prompt_id for item in prompts]
        if any(not prompt_id.strip() for prompt_id in prompt_ids):
            raise ValueError("prompt IDs must not be blank")
        if len(prompt_ids) != len(set(prompt_ids)):
            raise ValueError("prompt IDs must be unique")
        return values


def _model_json(model: Any) -> dict[str, Any]:
    model_dump = getattr(model, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    return json.loads(model.json())


def _decode_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) < 3 or not lines[-1].strip().startswith("```"):
            raise QwenContractError("unterminated JSON code fence")
        text = "\n".join(lines[1:-1]).strip()
        if text.lower().startswith("json"):
            text = text[4:].lstrip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise QwenContractError("Qwen response must be valid JSON") from exc
    if not isinstance(payload, dict):
        raise QwenContractError("Qwen response must be a JSON object")
    return payload


def parse_visual_facts(raw: str) -> QwenVisualFacts:
    try:
        return QwenVisualFacts(**_decode_json_object(raw))
    except QwenContractError:
        raise
    except Exception as exc:
        raise QwenContractError(
            "Qwen visual facts do not match the required schema"
        ) from exc


def parse_prompt_set(raw: str) -> QwenPromptSet:
    try:
        return QwenPromptSet(**_decode_json_object(raw))
    except QwenContractError:
        raise
    except Exception as exc:
        raise QwenContractError(
            "Qwen prompts do not satisfy the 3+2+1 contract"
        ) from exc


def build_visual_facts_messages(
    context: QwenVisualContext,
    *,
    images: list[QwenImageInput] | None = None,
) -> list[dict[str, Any]]:
    schema_example = {
        "target_object": "画面中央偏左的一名作业人员",
        "instance_count": 1,
        "visual_anchor": ["位于画面中央偏左", "穿深色上衣"],
        "mask_granularity": "人员整体",
        "visible_facts": ["人员头部没有可见安全帽"],
        "risk_semantics": "头部防护缺失",
    }
    user_text = (
        "请查看随消息提供的原图和目标框/mask，结合以下候选上下文，"
        "提取单一可分割目标的视觉事实。\n"
        f"候选上下文：{json.dumps(_model_json(context), ensure_ascii=False)}\n"
        "输出字段必须与此示例完全一致："
        f"{json.dumps(schema_example, ensure_ascii=False)}"
    )
    user_content: str | list[dict[str, Any]]
    if images:
        user_content = [{"type": "text", "text": user_text}]
        for image in images:
            user_content.extend(
                [
                    {
                        "type": "text",
                        "text": (
                            f"{image.label}。mask叠加图中的高亮像素决定"
                            "分割目标，原图用于识别对象和可见属性。"
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": image.data_url},
                    },
                ]
            )
    else:
        user_content = user_text
    return [
        {
            "role": "system",
            "content": (
                "你是施工安全图片的视觉事实提取器。只记录图片、目标框或"
                "mask中可以直接确认的事实；检测框和规则候选只是定位线索，"
                "不能当成事实。不得补充不可见动作、原因、法规结论或事故"
                "后果。mask存在时，目标对象、实例数量和粒度必须以mask实际"
                "覆盖像素为准。无法可靠识别时不要猜测。只输出一个JSON对象，"
                "不输出Markdown。"
            ),
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]


def build_prompt_enrichment_messages(
    *,
    category: AnnotationCategory,
    facts: QwenVisualFacts,
) -> list[dict[str, str]]:
    output_example = {
        "prompts": [
            {"prompt_id": "visual-1", "type": "visual", "text": "示例一"},
            {"prompt_id": "visual-2", "type": "visual", "text": "示例二"},
            {"prompt_id": "visual-3", "type": "visual", "text": "示例三"},
            {"prompt_id": "risk-1", "type": "risk", "text": "示例四"},
            {"prompt_id": "risk-2", "type": "risk", "text": "示例五"},
            {"prompt_id": "agent-1", "type": "agent", "text": "示例六"},
        ]
    }
    return [
        {
            "role": "system",
            "content": (
                "你是ReasonSeg训练Prompt生成器。输入事实已经过视觉提取。"
                "不得添加输入中不存在的对象、数量、位置、动作、原因或后果。"
                "六条Prompt必须指向同一目标、同一实例数量和同一mask粒度。"
                "视觉定位型、安全风险型和Agent查询型只能改变句式与业务表达，"
                "不能改变分割目标。每条文本必须在脱离上下文后仍能独立指向"
                "具体可分割对象。"
                "只输出JSON对象，不输出Markdown。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"来源类别：{category.value}\n"
                f"视觉事实：{json.dumps(_model_json(facts), ensure_ascii=False)}\n"
                f"类别语义边界：{RISK_SEMANTIC_BOUNDARIES[category]}\n"
                "生成恰好3条visual、2条risk、1条agent Prompt，文本互不"
                "重复。risk Prompt去掉风险描述后仍必须保留具体可分割对象。"
                "不得用危险区域、不安全目标、违规位置等抽象词代替具体对象。"
                "输出格式："
                f"{json.dumps(output_example, ensure_ascii=False)}"
            ),
        },
    ]
