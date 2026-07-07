from __future__ import annotations

from pathlib import Path

import yaml

from textSummarizer.grading.rubric import GradingRubric, RubricScore

DEFAULT_RUBRICS_DIR = Path(__file__).resolve().parents[3] / "config" / "rubrics"


def load_rubric_from_yaml(path: str | Path) -> GradingRubric:
    """Load a custom grading rubric from a YAML file."""
    rubric_path = Path(path)
    if rubric_path.suffix != ".yaml":
        rubric_path = rubric_path.with_suffix(".yaml")
    if not rubric_path.is_absolute() and not rubric_path.exists():
        rubric_path = DEFAULT_RUBRICS_DIR / rubric_path.name
    with rubric_path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    dimensions = tuple(data.get("dimensions", RubricScore.DIMENSIONS))
    return GradingRubric(
        dimensions=dimensions,
        threshold=float(data.get("threshold", 3.5)),
        descriptions=dict(data.get("descriptions", {})),
    )


def list_rubrics(rubrics_dir: str | Path | None = None) -> list[str]:
    """List available YAML rubric files."""
    directory = Path(rubrics_dir) if rubrics_dir else DEFAULT_RUBRICS_DIR
    if not directory.exists():
        return []
    return sorted(path.stem for path in directory.glob("*.yaml"))
