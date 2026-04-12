from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from langchain.agents import create_agent
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import CRITIC_SYSTEM_PROMPT, Settings, preview_for_log
from schemas import CritiqueResult


logger = logging.getLogger(__name__)


async def critique_findings_json_async(findings: str, tools: list[BaseTool]) -> str:
    settings = Settings()
    llm = ChatOpenAI(
        api_key=settings.api_key.get_secret_value(),
        model=settings.critic_model_name,
        timeout=settings.request_timeout,
    )
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=CRITIC_SYSTEM_PROMPT,
        response_format=CritiqueResult,
    )

    logger.info(
        "CriticAgent: invoke start findings_chars=%d preview=%s",
        len(findings),
        preview_for_log(findings, 300),
    )
    result = await agent.ainvoke({"messages": [("user", findings)]})
    structured = result.get("structured_response")
    if isinstance(structured, CritiqueResult):
        out = structured
    elif hasattr(structured, "model_dump"):
        out = CritiqueResult.model_validate(structured.model_dump())
    elif isinstance(structured, dict):
        out = CritiqueResult.model_validate(structured)
    else:
        logger.warning("CriticAgent: structured_response missing; using fallback critique")
        out = CritiqueResult(
            verdict="REVISE",
            is_fresh=False,
            is_complete=False,
            is_well_structured=False,
            strengths=[],
            gaps=["Critic could not produce structured response"],
            revision_requests=["Run additional verification and provide complete findings"],
        )

    logger.info("CriticAgent: invoke end verdict=%s", out.verdict)
    return json.dumps(out.model_dump(), ensure_ascii=False, indent=2)
