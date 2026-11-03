"""Run the evaluation cases against the live agent and save a report.

    uv run car-agent-eval                 # all cases
    uv run car-agent-eval --limit 3       # first 3 cases
    uv run car-agent-eval --model openai/gpt-oss-20b
    uv run car-agent-eval --rescore eval_results/run-XXXX.json   # recompute metrics, no LLM calls
"""
from __future__ import annotations

import argparse
import json
import re
import time
import uuid
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

from car_agent.config import PROJECT_ROOT, load_settings
from car_agent.evaluation.metrics import score_turn
from car_agent.graph import build_agent
from car_agent.prompts import system_prompt
from car_agent.tool_schemas import tool_schemas

CASES_PATH = Path(__file__).with_name("cases.json")
METRICS = ["tool_recall", "tool_precision", "args_accuracy", "grounding"]


def resolve_placeholders(value: Any, mapping: dict[str, Any]) -> Any:
    """Replace "{name}" placeholders: whole-string ones by the raw value, inline ones by its text."""
    if isinstance(value, str):
        if value in mapping:
            return mapping[value]
        for key, replacement in mapping.items():
            value = value.replace(key, str(replacement))
        return value
    if isinstance(value, list):
        return [resolve_placeholders(item, mapping) for item in value]
    if isinstance(value, dict):
        return {key: resolve_placeholders(item, mapping) for key, item in value.items()}
    return value


def placeholder_values(listings) -> dict[str, Any]:
    good = listings.loc[listings["deal_verdict"] == "good_deal", "id"]
    return {"{good_deal_id}": int(good.iloc[0] if len(good) else listings["id"].iloc[0])}


def markdown_report(rows: list[dict], summary: dict, model: str) -> str:
    lines = [
        f"# Agent evaluation - {summary['timestamp']}",
        "",
        f"Model: `{model}` · cases: {summary['cases']} · passed: {summary['passed']} "
        f"({summary['pass_rate']:.0%}) · mean time per case: {summary['mean_seconds']:.1f}s",
        "",
        "| Metric | Mean |",
        "|---|---|",
        *[f"| {name} | {summary[name]:.3f} |" for name in METRICS + ["language_match"]],
        "",
        "| Case | Passed | Tools called | Recall | Precision | Args | Grounding | Language | Problems |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        s = row.get("score")
        if s is None:
            lines.append(f"| {row['id']} | error | | | | | | | {row['error']} |")
            continue
        problems = []
        if s["missing_tools"]:
            problems.append("missing: " + ", ".join(s["missing_tools"]))
        if s["failed_arg_checks"]:
            problems.append("args: " + ", ".join(s["failed_arg_checks"]))
        if s["ungrounded_numbers"]:
            problems.append("ungrounded: " + ", ".join(f"{n:g}" for n in s["ungrounded_numbers"][:5]))
        lines.append(
            f"| {row['id']} | {'yes' if s['passed'] else 'no'} | {', '.join(s['called_tools'])} | "
            f"{s['tool_recall']:.2f} | {s['tool_precision']:.2f} | {s['args_accuracy']:.2f} | "
            f"{s['grounding']:.2f} | {'yes' if s['language_match'] else 'no'} | {'; '.join(problems)} |"
        )
    return "\n".join(lines) + "\n"


def reference_texts(data_snapshot: str | None, max_results: int) -> tuple[str, ...]:
    """What the agent knows besides tool results: its instructions and the tool descriptions."""
    return system_prompt(data_snapshot), json.dumps(tool_schemas(max_results))


def summarize(rows: list[dict], model: str) -> dict:
    scored = [row["score"] for row in rows if "score" in row]
    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "model": model,
        "cases": len(rows),
        "errors": len(rows) - len(scored),
        "passed": sum(s["passed"] for s in scored),
        "pass_rate": sum(s["passed"] for s in scored) / len(rows) if rows else 0.0,
        "mean_seconds": sum(row.get("seconds", 0) for row in rows) / max(1, len(scored)),
        **{name: sum(s[name] for s in scored) / max(1, len(scored)) for name in METRICS},
        "language_match": sum(s["language_match"] for s in scored) / max(1, len(scored)),
    }


def save_report(rows: list[dict], summary: dict, out_dir: Path, prefix: str = "run") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"{prefix}-{datetime.now():%Y%m%d-%H%M%S}"
    stem.with_suffix(".json").write_text(
        json.dumps({"summary": summary, "cases": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    report = markdown_report(rows, summary, summary["model"])
    stem.with_suffix(".md").write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"Saved: {stem}.json and {stem}.md")


def _case_for_saved_row(template: dict, saved_turns: list[str]) -> dict:
    """Recover placeholder values (e.g. the listing id) of an old run by matching template turns."""
    mapping: dict[str, Any] = {}
    for pattern, actual in zip(template["turns"], saved_turns):
        for name in re.findall(r"\{\w+\}", pattern):
            regex = re.escape(pattern).replace(re.escape(name), r"(\d+)")
            match = re.fullmatch(regex, actual)
            if match:
                mapping[name] = int(match.group(1))
    return resolve_placeholders(template, mapping)


def rescore(run_path: Path, cases_path: Path, out_dir: Path) -> None:
    """Recompute metrics for a saved run (no LLM calls), e.g. after changing the metrics."""
    saved = json.loads(run_path.read_text(encoding="utf-8"))
    templates = {case["id"]: case for case in json.loads(cases_path.read_text(encoding="utf-8"))}
    settings = load_settings()
    references = reference_texts(None, settings.max_results)
    rows = []
    for row in saved["cases"]:
        if "answer" in row and row["id"] in templates:
            # Always the current case definition, so that new checks also apply to old runs.
            case = _case_for_saved_row(templates[row["id"]], row["turns"])
            row = {**row, "case": case,
                   "score": score_turn(case, row["answer"], row["tool_calls"], row["turns"], references).to_dict()}
        rows.append(row)
    summary = {**summarize(rows, saved["summary"]["model"]), "rescored_from": run_path.name}
    save_report(rows, summary, out_dir, prefix="rescore")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the car price agent.")
    parser.add_argument("--cases", type=Path, default=CASES_PATH)
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N cases")
    parser.add_argument("--model", default=None, help="override GROQ_MODEL")
    parser.add_argument("--pause", type=float, default=2.0, help="seconds between cases (free-tier limits)")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "eval_results")
    parser.add_argument("--rescore", type=Path, default=None,
                        help="recompute metrics for a saved run JSON instead of calling the agent")
    args = parser.parse_args()

    if args.rescore:
        rescore(args.rescore, args.cases, args.out)
        return

    settings = load_settings()
    if args.model:
        settings = replace(settings, groq_model=args.model)
    agent, store = build_agent(settings)
    references = reference_texts(store.data_snapshot, settings.max_results)

    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    cases = resolve_placeholders(cases, placeholder_values(store.listings))[: args.limit]

    rows = []
    for index, case in enumerate(cases, 1):
        thread_id = f"eval-{case['id']}-{uuid.uuid4().hex[:8]}"
        print(f"[{index}/{len(cases)}] {case['id']} ...", flush=True)
        try:
            for earlier in case["turns"][:-1]:
                agent.ask(earlier, thread_id)
            turn = agent.ask(case["turns"][-1], thread_id)
            calls = [{"name": c.name, "args": c.args, "output": c.output} for c in turn.tool_calls]
            score = score_turn(case, turn.answer, calls, case["turns"], references)
            rows.append({"id": case["id"], "case": case, "turns": case["turns"], "answer": turn.answer,
                         "tool_calls": calls, "seconds": round(turn.seconds, 1), "score": score.to_dict()})
            print(f"    passed={score.passed} tools={score.called_tools} grounding={score.grounding} "
                  f"({turn.seconds:.1f}s)")
        except Exception as error:  # one failing case must not stop the run
            rows.append({"id": case["id"], "turns": case["turns"], "error": f"{type(error).__name__}: {error}"})
            print(f"    ERROR {type(error).__name__}: {error}")
        if index < len(cases):
            time.sleep(args.pause)

    save_report(rows, summarize(rows, settings.groq_model), args.out)


if __name__ == "__main__":
    main()
