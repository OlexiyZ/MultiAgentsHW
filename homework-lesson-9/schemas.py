from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ResearchPlan(BaseModel):
    """Decomposes the user's request into a goal, searchable queries, sources to use, and expected report shape."""

    goal: str = Field(default="", description="What we are trying to answer.")
    search_queries: list[str] = Field(default_factory=list, description="Specific queries to execute.")
    sources_to_check: list[str] = Field(
        default_factory=lambda: ["knowledge_base", "web"],
        description="'knowledge_base', 'web', or both.",
    )
    output_format: str = Field(
        default="Markdown report with headings and sources",
        description="What the final report should look like.",
    )


class CritiqueResult(BaseModel):
    """Structured evaluation of research with an approval verdict."""

    verdict: Literal["APPROVE", "REVISE"] = Field(description="APPROVE or REVISE.")
    is_fresh: bool = Field(description="Is the data up-to-date?")
    is_complete: bool = Field(description="Does research cover the user's request?")
    is_well_structured: bool = Field(description="Are findings report-ready?")
    strengths: list[str] = Field(description="What is good about the research.")
    gaps: list[str] = Field(description="What is missing or outdated.")
    revision_requests: list[str] = Field(description="Specific fixes if verdict is REVISE.")
