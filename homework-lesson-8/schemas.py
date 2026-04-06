from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ResearchPlan(BaseModel):
    """Decomposes the user's request into a goal, searchable queries, sources to use, and expected report shape.
    Розбиває запит користувача на мету, пошукові запити, джерела та очікуваний формат звіту."""

    goal: str = Field(
        description=(
            "What we are trying to answer. "
            # "На що саме має відповісти дослідження."
        )
    )
    search_queries: list[str] = Field(
        description=(
            "Specific queries to execute. "
            # "Конкретні пошукові запити для виконання."
        )
    )
    sources_to_check: list[str] = Field(
        description=(
            "Sources to use: 'knowledge_base', 'web', or both. "
            # "Джерела для перевірки: 'knowledge_base', 'web' або обидва."
        )
    )
    output_format: str = Field(
        description=(
            "What the final report should look like. "
            # "Як має виглядати фінальний звіт."
        )
    )


class CritiqueResult(BaseModel):
    """Structured evaluation of research with an approval verdict and notes on freshness, completeness, and organization.
    Структурована оцінка дослідження з вердиктом і зауваженнями щодо актуальності, повноти та структури."""

    verdict: Literal["APPROVE", "REVISE"] = Field(
        description=(
            "APPROVE if research is acceptable; REVISE if more work is needed. "
            # "APPROVE — прийняти; REVISE — потрібне доопрацювання."
        )
    )
    is_fresh: bool = Field(
        description=(
            "Is the data up-to-date and based on recent sources? "
            # "Чи дані актуальні й ґрунтуються на недавніх джерелах?"
        )
    )
    is_complete: bool = Field(
        description=(
            "Does the research fully cover the user's original request? "
            # "Чи дослідження повністю покриває початковий запит користувача?"
        )
    )
    is_well_structured: bool = Field(
        description=(
            "Are findings logically organized and ready for a report? "
            # "Чи логічно структуровані знахідки та чи готові для звіту?"
        )
    )
    strengths: list[str] = Field(
        description=(
            "What is good about the research. "
            # "Що добре в дослідженні."
        )
    )
    gaps: list[str] = Field(
        description=(
            "What is missing, outdated, or poorly structured. "
            # "Чого бракує, що застаріло або погано структуровано."
        )
    )
    revision_requests: list[str] = Field(
        description=(
            "Specific things to fix if verdict is REVISE. "
            # "Що саме виправити, якщо вердикт REVISE."
        )
    )
