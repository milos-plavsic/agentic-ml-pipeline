from pathlib import Path

from analysis.report import generate_report


def test_generate_report_smoke(tmp_path: Path) -> None:
    out = generate_report(tmp_path, random_state=0)
    assert "output_dir" in out
    assert (tmp_path / "summary.json").is_file()
    assert (tmp_path / "figures" / "actual_vs_predicted.png").is_file()
    raw = (tmp_path / "summary.json").read_text(encoding="utf-8")
    assert "NaN" not in raw  # strict JSON, not Python NaN token
