from __future__ import annotations

import json

from tsuzuri.cli import main
from tsuzuri.evaluation import load_cases, run_case, run_recall_eval


def test_bundled_eval_cases_are_well_formed():
    cases = load_cases()
    assert len(cases) >= 10
    for case in cases:
        assert case["id"] and case["query"]
        assert case.get("expect_any") or case.get("expect_none")


def test_isolation_cases_pass_offline(tmp_path):
    cases = [c for c in load_cases() if c["id"].startswith("isolation-")]
    result = run_recall_eval(cases=cases, data_dir=str(tmp_path))
    assert result["summary"]["passed"] == len(cases)


def test_single_case_reports_hits(tmp_path):
    case = {"id": "fact-pet", "agent": "叶澄", "turns": ["我家有只橘猫，叫团子"],
            "query": "我家猫叫什么名字？", "expect_any": ["团子"]}
    row = run_case(case, data_dir=str(tmp_path))
    assert row["ok"] and row["hit"] == ["团子"]


def test_cli_demo_and_recall(tmp_path, capsys):
    assert main(["--data-dir", str(tmp_path / "demo"), "demo"]) == 0
    out = capsys.readouterr().out
    assert "data dir" in out
    assert main(["--data-dir", str(tmp_path / "demo"), "recall", "面试"]) == 0


def test_cli_eval_writes_report(tmp_path):
    report = tmp_path / "report.json"
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text(json.dumps({"id": "fact-pet", "agent": "叶澄", "turns": ["我家有只橘猫，叫团子"],
                                   "query": "我家猫叫什么名字？", "expect_any": ["团子"]}, ensure_ascii=False) + "\n",
                       encoding="utf-8")
    assert main(["eval", "--dataset", str(dataset), "--report", str(report)]) == 0
    assert json.loads(report.read_text(encoding="utf-8"))["summary"]["cases"] == 1
