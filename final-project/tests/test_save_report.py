from __future__ import annotations

from pathlib import Path

import tools


def test_safe_report_path_uses_topic_slug_and_versions(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(tools, "BASE_DIR", tmp_path)
    monkeypatch.setattr(tools.settings, "output_dir", "output")

    first = tools._safe_report_path("Максимальний термін дії консенту")
    assert first.name == "maksymalnyi-termin-dii-konsentu.md"

    first.parent.mkdir(parents=True, exist_ok=True)
    first.write_text("v1", encoding="utf-8")

    second = tools._safe_report_path("Максимальний термін дії консенту")
    assert second.name == "maksymalnyi-termin-dii-konsentu.v2.md"


def test_save_report_tool_accepts_topic_not_filename(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(tools, "BASE_DIR", tmp_path)
    monkeypatch.setattr(tools.settings, "output_dir", "output")

    result = tools.save_report.invoke(
        {
            "topic": "Open banking consent",
            "content": "# Report\n\nBody",
        }
    )

    target = tmp_path / "output" / "open-banking-consent.md"
    assert target.read_text(encoding="utf-8") == "# Report\n\nBody"
    assert str(target) in result
