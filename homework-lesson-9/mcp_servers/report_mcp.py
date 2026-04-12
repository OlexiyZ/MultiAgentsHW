"""ReportMCP: save_report + output-dir listing resource."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from fastmcp import FastMCP

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import BASE_DIR, Settings, preview_for_log  # noqa: E402

logger = logging.getLogger(__name__)
settings = Settings()
mcp = FastMCP("ReportMCP")


def _safe_report_path(filename: str) -> Path:
    candidate = Path(filename)
    safe_name = candidate.name or "report.md"
    if not safe_name.lower().endswith(".md"):
        safe_name = f"{safe_name}.md"

    output_dir = BASE_DIR / settings.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / safe_name


@mcp.tool()
def save_report(filename: str, content: str) -> str:
    """Write Markdown report under output/ and return status text."""

    logger.info(
        "MCP save_report: filename=%r content_chars=%d preview=%s",
        filename,
        len(content),
        preview_for_log(content, 240),
    )
    try:
        target = _safe_report_path(filename)
        target.write_text(content, encoding="utf-8")
        logger.info("MCP save_report: wrote path=%s", target)
        return f"Report saved to {target}"
    except Exception as exc:
        logger.exception("MCP save_report: failed")
        return f"Failed to save report: {exc}"


@mcp.resource("resource://output-dir")
def output_dir_resource() -> str:
    """Configured output directory path and list of saved .md reports."""

    d = BASE_DIR / settings.output_dir
    d.mkdir(parents=True, exist_ok=True)
    files = sorted(p.name for p in d.glob("*.md"))
    payload = {"path": str(d.resolve()), "reports": files}
    return json.dumps(payload, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    port = settings.report_mcp_port
    mcp.run(transport="http", host="0.0.0.0", port=port)
