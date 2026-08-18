from __future__ import annotations

from fastapi import APIRouter, Depends

from ..auth import AuthDependency
from ..schemas import DefaultQwenPromptTemplateResponse, ErrorPayload
from ...pipeline.qwen.contract import (
    DEFAULT_QWEN_PROMPT_CONTEXT_PLACEHOLDER,
    DEFAULT_QWEN_PROMPT_TEMPLATE,
    DEFAULT_QWEN_PROMPT_TEMPLATE_ID,
)


def build_prompt_templates_router(
    *,
    authenticate: AuthDependency,
) -> APIRouter:
    router = APIRouter(
        prefix="/v1/annotation/prompt-templates",
        tags=["Prompt Templates"],
    )

    @router.get(
        "/default",
        operation_id="getDefaultQwenPromptTemplate",
        response_model=DefaultQwenPromptTemplateResponse,
        dependencies=[Depends(authenticate)],
        responses={
            401: {
                "model": ErrorPayload,
                "description": "Authentication failed.",
            },
        },
    )
    async def get_default_prompt_template() -> dict[str, str]:
        return {
            "template_id": DEFAULT_QWEN_PROMPT_TEMPLATE_ID,
            "template": DEFAULT_QWEN_PROMPT_TEMPLATE,
            "context_placeholder": (
                DEFAULT_QWEN_PROMPT_CONTEXT_PLACEHOLDER
            ),
        }

    return router
