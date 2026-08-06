from __future__ import annotations

import json
from dataclasses import dataclass
from collections import Counter
from typing import Any, List, Literal, Optional

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
QWEN_JOINT_FACTS_PROMPT_VERSION = "construction-joint-visible-facts-v2"
QWEN_JOINT_ENRICHMENT_PROMPT_VERSION = (
    "construction-joint-prompts-3-2-1-grounded-v6"
)

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


class QwenJointTarget(StrictModel):
    task_id: str
    task_version: int = Field(..., ge=1)
    category: AnnotationCategory
    candidate_target_object: str = Field(
        ...,
        min_length=1,
        max_length=300,
    )
    candidate_detection_entity: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=200,
    )
    target_box_xyxy: List[float]
    target_detection_ids: List[str] = Field(default_factory=list)

    @validator(
        "task_id",
        "candidate_target_object",
        "candidate_detection_entity",
        pre=True,
    )
    def joint_target_text_is_not_blank(cls, value):
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("joint target text must not be blank")
        return normalized

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

    @validator("target_detection_ids")
    def detection_ids_are_unique(cls, value: List[str]) -> List[str]:
        normalized = [item.strip() for item in value]
        if any(not item for item in normalized):
            raise ValueError("target_detection_ids must not be blank")
        if len(normalized) != len(set(normalized)):
            raise ValueError("target_detection_ids must be unique")
        return normalized


class QwenJointVisualContext(StrictModel):
    asset_id: str
    targets: List[QwenJointTarget] = Field(
        ...,
        min_items=2,
        max_items=16,
    )
    requires_visual_verification: bool = True
    all_masks_available: bool = True
    all_crops_available: bool = True

    @validator("targets")
    def task_ids_are_unique(cls, value: List[QwenJointTarget]):
        task_ids = [item.task_id for item in value]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("joint target task IDs must be unique")
        return value


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


class QwenJointFactTarget(StrictModel):
    task_id: str
    target_object: str = Field(..., min_length=1, max_length=300)
    instance_count: int = Field(..., ge=1)
    visual_anchor: List[str] = Field(..., min_items=1, max_items=10)

    @validator("task_id", "target_object", pre=True)
    def text_is_not_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("joint fact target text must not be blank")
        return normalized

    @validator("visual_anchor")
    def anchors_are_non_empty_and_unique(
        cls,
        value: List[str],
    ) -> List[str]:
        normalized = [item.strip() for item in value]
        if any(not item for item in normalized):
            raise ValueError("joint target anchors must not be blank")
        if len(normalized) != len(set(normalized)):
            raise ValueError("joint target anchors must be unique")
        return normalized


class QwenJointVisualFacts(QwenVisualFacts):
    task_targets: List[QwenJointFactTarget] = Field(
        ...,
        min_items=2,
        max_items=16,
    )

    @validator("task_targets")
    def task_target_ids_are_unique(
        cls,
        value: List[QwenJointFactTarget],
    ) -> List[QwenJointFactTarget]:
        task_ids = [item.task_id for item in value]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("joint fact Task IDs must be unique")
        return value


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


class QwenJointPromptEnvelope(StrictModel):
    covered_task_ids: List[str] = Field(..., min_items=2, max_items=16)
    fact_consistent: Literal[True]
    prompts: List[AnnotationPrompt] = Field(
        ...,
        min_items=6,
        max_items=6,
    )

    @validator("covered_task_ids")
    def covered_task_ids_are_non_empty_and_unique(
        cls,
        value: List[str],
    ) -> List[str]:
        normalized = [item.strip() for item in value]
        if any(not item for item in normalized):
            raise ValueError("covered Task IDs must not be blank")
        if len(normalized) != len(set(normalized)):
            raise ValueError("covered Task IDs must be unique")
        return normalized


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


def parse_joint_visual_facts(
    raw: str,
    *,
    expected_task_ids: List[str],
) -> QwenJointVisualFacts:
    try:
        facts = QwenJointVisualFacts(**_decode_json_object(raw))
    except QwenContractError:
        raise
    except Exception as exc:
        raise QwenContractError(
            "Qwen joint visual facts do not match the required schema"
        ) from exc
    actual_task_ids = [item.task_id for item in facts.task_targets]
    if actual_task_ids != expected_task_ids:
        raise QwenContractError(
            "Qwen joint visual facts did not cover every Task in order"
        )
    return facts


def parse_prompt_set(raw: str) -> QwenPromptSet:
    try:
        return QwenPromptSet(**_decode_json_object(raw))
    except QwenContractError:
        raise
    except Exception as exc:
        raise QwenContractError(
            "Qwen prompts do not satisfy the 3+2+1 contract"
        ) from exc


def parse_joint_prompt_set(
    raw: str,
    *,
    expected_task_ids: List[str],
) -> QwenPromptSet:
    try:
        envelope = QwenJointPromptEnvelope(
            **_decode_json_object(raw)
        )
        prompt_set = QwenPromptSet(prompts=envelope.prompts)
    except QwenContractError:
        raise
    except Exception as exc:
        raise QwenContractError(
            "Qwen joint prompts do not satisfy the coverage contract"
        ) from exc
    if envelope.covered_task_ids != expected_task_ids:
        raise QwenContractError(
            "Qwen joint prompts did not cover every Task in order"
        )
    return prompt_set


def ground_joint_prompt_set(
    *,
    facts: QwenJointVisualFacts,
) -> QwenPromptSet:
    target = facts.target_object[:100].rstrip("，。； ")
    visible_fact = facts.visible_facts[0][:70].rstrip("，。； ")
    anchor = facts.visual_anchor[0][:70].rstrip("，。； ")
    mask_granularity = facts.mask_granularity[:60].rstrip("，。； ")
    grounded = [
        AnnotationPrompt(
            prompt_id="visual-1",
            type=PromptType.VISUAL,
            text=f"分割{target}。",
        ),
        AnnotationPrompt(
            prompt_id="visual-2",
            type=PromptType.VISUAL,
            text=f"标出{target}；关系锚点：{anchor}。",
        ),
        AnnotationPrompt(
            prompt_id="visual-3",
            type=PromptType.VISUAL,
            text=(
                f"提取{mask_granularity}对应的{target}；"
                f"可见事实：{visible_fact}。"
            ),
        ),
        AnnotationPrompt(
            prompt_id="risk-1",
            type=PromptType.RISK,
            text=(
                f"从安全标注角度分割{target}；"
                f"可见事实：{visible_fact}。"
            ),
        ),
        AnnotationPrompt(
            prompt_id="risk-2",
            type=PromptType.RISK,
            text=(
                f"标出{target}，保持{mask_granularity}；"
                f"关系锚点：{anchor}。"
            ),
        ),
        AnnotationPrompt(
            prompt_id="agent-1",
            type=PromptType.AGENT,
            text=(
                f"请定位并联合分割{target}，"
                "保留全部所选Task对应的mask目标。"
            ),
        ),
    ]
    try:
        return QwenPromptSet(prompts=grounded)
    except Exception as exc:
        raise QwenContractError(
            "grounded joint prompts exceed the annotation contract"
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


def build_joint_visual_facts_messages(
    context: QwenJointVisualContext,
    *,
    images: list[QwenImageInput] | None = None,
) -> list[dict[str, Any]]:
    task_examples = [
        {
            "task_id": target.task_id,
            "target_object": target.candidate_target_object,
            "instance_count": 1,
            "visual_anchor": ["按该Task的mask确认目标位置和外观"],
        }
        for target in context.targets
    ]
    schema_example = {
        "target_object": "画面中的一名作业人员及其穿着的反光背心",
        "instance_count": 2,
        "visual_anchor": ["反光背心位于人员躯干处", "两者属于穿戴关系"],
        "mask_granularity": "所选人员和反光背心的整体联合mask",
        "visible_facts": ["人员穿着反光背心"],
        "risk_semantics": "人员可见性防护状态以画面事实为准",
        "task_targets": task_examples,
    }
    user_text = (
        "请查看同一张原图、每个所选目标的mask以及对应裁剪图，提取这些"
        "目标作为一个整体时的视觉事实和相互关系。每个Task代表一个独立"
        "目标，即使一个目标是另一个目标的组成部分，也必须分别识别并共同"
        "描述。候选对象名和检测实体仅用于定位，最终事实仍以图像和mask为准。"
        "task_targets必须按输入顺序逐项返回全部Task ID；每项分别描述该mask"
        "覆盖的对象和实例数。总instance_count表示联合集合中的目标实体总数，"
        "不能理解成某一种对象（例如人员）的数量。\n"
        f"Task Group上下文："
        f"{json.dumps(_model_json(context), ensure_ascii=False)}\n"
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
                            f"{image.label}。各Task的mask决定对应目标像素，"
                            "原图用于判断目标之间可见的方位、距离和关系。"
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
                "你是施工安全图片的多目标联合视觉事实提取器。只能记录"
                "原图、各目标mask和裁剪图可以直接确认的事实。必须识别并"
                "共同描述全部所选目标，不得遗漏任一Task，也不得把多个"
                "目标误写成单一目标。重点提取目标之间可见的空间或业务"
                "关系；无法确认的动作、因果、违规状态或风险不得猜测。"
                "如果一个mask是人员、另一个mask是该人员穿戴的安全帽或"
                "反光背心，应明确写出人员与防护用品的穿戴/从属关系，不得"
                "只写人员，也不得把两个Task误写成两个人。每个task_targets"
                "元素只能解释对应Task的mask；Task ID必须原样抄写且顺序不变。"
                "所有mask像素的集合决定最终联合分割范围。只输出一个JSON"
                "对象，不输出Markdown。"
            ),
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]


def build_joint_prompt_enrichment_messages(
    *,
    categories: List[AnnotationCategory],
    facts: QwenJointVisualFacts,
) -> list[dict[str, str]]:
    task_ids = [item.task_id for item in facts.task_targets]
    prompt_facts = _model_json(facts)
    prompt_facts.pop("instance_count", None)
    output_example = {
        "covered_task_ids": task_ids,
        "fact_consistent": True,
        "prompts": [
            {"prompt_id": "visual-1", "type": "visual", "text": "示例一"},
            {"prompt_id": "visual-2", "type": "visual", "text": "示例二"},
            {"prompt_id": "visual-3", "type": "visual", "text": "示例三"},
            {"prompt_id": "risk-1", "type": "risk", "text": "示例四"},
            {"prompt_id": "risk-2", "type": "risk", "text": "示例五"},
            {"prompt_id": "agent-1", "type": "agent", "text": "示例六"},
        ]
    }
    unique_categories = list(dict.fromkeys(categories))
    boundaries = {
        category.value: RISK_SEMANTIC_BOUNDARIES[category]
        for category in unique_categories
    }
    return [
        {
            "role": "system",
            "content": (
                "你是ReasonSeg多目标联合Prompt生成器。每条Prompt都必须"
                "同时指向输入事实中的全部目标及其可见关系，对应所有成员"
                "mask的联合像素。不得只描述其中一个目标，也不得新增输入"
                "事实中不存在的对象、数量、位置、动作、原因或后果。六条"
                "Prompt必须保持完全相同的目标集合、实例数量、关系和mask"
                "粒度，只能改变句式与业务表达。必须逐项使用task_targets"
                "中的对象与实例数；总instance_count不是某一对象类别的数量。"
                "来源类别只是路由元数据，不是视觉事实；类别语义边界只限制"
                "可用措辞，不能证明违规状态。如果事实显示已佩戴、已穿着或"
                "符合要求，任何Prompt都不得反写成未佩戴、未穿着或违规。"
                "risk Prompt在没有可见风险时应描述已确认的防护状态或中性"
                "安全意义，不能虚构隐患，也不能添加光照、环境、事故概率、"
                "降低风险等联合视觉事实中未明确出现的条件或后果。"
                "只输出JSON对象，不输出Markdown。"
            ),
        },
        {
            "role": "user",
            "content": (
                "来源类别："
                f"{json.dumps([item.value for item in unique_categories], ensure_ascii=False)}\n"
                f"联合视觉事实："
                f"{json.dumps(prompt_facts, ensure_ascii=False)}\n"
                "类别语义边界："
                f"{json.dumps(boundaries, ensure_ascii=False)}\n"
                "生成恰好3条visual、2条risk、1条agent Prompt，文本互不"
                "重复。每一条都必须明确包含全部所选对象和它们之间由事实"
                "支持的关系；不得退化成若干互不相关的单目标描述。"
                "covered_task_ids必须原样按顺序返回全部Task ID；只有逐条"
                "检查六条Prompt均未遗漏目标且未反转事实后，fact_consistent"
                "才能输出true。输出格式："
                f"{json.dumps(output_example, ensure_ascii=False)}"
            ),
        },
    ]


def build_joint_prompt_review_messages(
    *,
    facts: QwenJointVisualFacts,
    candidate_prompts: QwenPromptSet,
) -> list[dict[str, str]]:
    task_ids = [item.task_id for item in facts.task_targets]
    review_facts = _model_json(facts)
    review_facts.pop("instance_count", None)
    output_example = {
        "covered_task_ids": task_ids,
        "fact_consistent": True,
        "prompts": [
            {"prompt_id": "visual-1", "type": "visual", "text": "示例一"},
            {"prompt_id": "visual-2", "type": "visual", "text": "示例二"},
            {"prompt_id": "visual-3", "type": "visual", "text": "示例三"},
            {"prompt_id": "risk-1", "type": "risk", "text": "示例四"},
            {"prompt_id": "risk-2", "type": "risk", "text": "示例五"},
            {"prompt_id": "agent-1", "type": "agent", "text": "示例六"},
        ],
    }
    return [
        {
            "role": "system",
            "content": (
                "你是多目标联合Prompt的事实一致性审核和修订器。必须逐条"
                "核对候选Prompt与task_targets，不是只做评价。发现对象遗漏、"
                "对象新增、实例数量错误、关系错误或安全状态反转时，必须直接"
                "改写对应Prompt。不同Task可以是人员及其穿戴物；两个Task不"
                "等于两个人。每个对象的数量只允许来自对应task_targets中的"
                "instance_count，不得把Task数量相加后套到某一对象类别。"
                "六条Prompt都必须同时描述全部Task目标及已确认关系。事实显示"
                "已穿戴或合规时，禁止改写成未穿戴、违规或存在隐患。完成逐条"
                "修订后才可输出fact_consistent=true。risk和agent文本仍是"
                "分割目标描述，不是安全科普；必须删除事实中未出现的光线不足、"
                "工作环境推断、事故概率、提升安全性、降低或减少风险等通用"
                "条件和后果。只能使用task_targets、visible_facts、"
                "visual_anchor和risk_semantics明确提供的事实。只输出JSON对象。"
            ),
        },
        {
            "role": "user",
            "content": (
                "逐Task视觉事实："
                f"{json.dumps(review_facts, ensure_ascii=False)}\n"
                "待审核候选Prompt："
                f"{json.dumps(_model_json(candidate_prompts), ensure_ascii=False)}\n"
                "保持3条visual、2条risk、1条agent且文本互不重复。"
                "covered_task_ids必须原样按顺序返回全部Task ID。输出格式："
                f"{json.dumps(output_example, ensure_ascii=False)}"
            ),
        },
    ]
