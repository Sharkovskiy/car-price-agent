"""Deterministic metrics for one agent turn (no LLM judge needed).

tool_recall     share of expected tools that were called
tool_precision  share of called tools that were expected or explicitly allowed
args_accuracy   share of argument checks passed: expected arguments present in at least one call
                of the tool, forbidden arguments present in none
grounding       share of numbers in the answer that can be traced to tool results, tool arguments
                or the user's own messages (catches invented prices and statistics)
language_match  the answer is written in the language of the question
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from car_agent.tools import MANUFACTURER_ALIASES, US_STATES

# Integer part with optional thousands separators, optional decimals, optional "k" / "тис." multiplier.
NUMBER = re.compile(
    r"(?<![\w.])(\d{1,3}(?:[   ,]\d{3})+|\d+)(?:[.,](\d+))?(?:\s?(k|K|тис\.?|thousand))?"
)
URL = re.compile(r"https?://\S+")
CYRILLIC = re.compile(r"[Ѐ-ӿ]")
LATIN = re.compile(r"[A-Za-z]")

MIN_CHECKED_NUMBER = 10        # small counts ("3 listings") are not checked
RELATIVE_TOLERANCE = 0.02      # rounding in the answer, e.g. 18 285 -> "about 18.3k"
GROUNDING_PASS = 0.9
# Numbers that are allowed without a source: the percentage base in "deal % = ... x 100".
ALWAYS_GROUNDED = (100.0,)


def extract_numbers(text: str) -> list[float]:
    numbers = []
    for integer, decimals, multiplier in NUMBER.findall(text or ""):
        value = float(re.sub(r"[   ,]", "", integer) + (f".{decimals}" if decimals else ""))
        if multiplier:
            value *= 1_000
        numbers.append(value)
    return numbers


def _numbers_in_json(value: Any) -> list[float]:
    if isinstance(value, bool) or value is None:
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, str):
        return extract_numbers(value)
    if isinstance(value, dict):
        # Keys count too: "p75" justifies "75th percentile" in the answer.
        return [n for key, item in value.items()
                for n in [float(d) for d in re.findall(r"\d+", str(key))] + _numbers_in_json(item)]
    if isinstance(value, list):
        return [n for item in value for n in _numbers_in_json(item)]
    return []


def numbers_in_tool_text(text: str) -> list[float]:
    try:
        return _numbers_in_json(json.loads(text))
    except ValueError:
        return extract_numbers(text)


def _close(a: float, b: float) -> bool:
    # Signs are ignored: answers write "-35 %" with typographic dashes that are not parsed as minus.
    a, b = abs(a), abs(b)
    return abs(a - b) <= max(1.0, RELATIVE_TOLERANCE * max(a, b))


def detect_language(text: str) -> str:
    text = URL.sub("", text or "")
    cyrillic, latin = len(CYRILLIC.findall(text)), len(LATIN.findall(text))
    return "uk" if cyrillic > 0.3 * max(1, cyrillic + latin) else "en"


def _normalize(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip().lower()
        text = US_STATES.get(text, text)
        return MANUFACTURER_ALIASES.get(text, text)
    return value


def _check_arg(actual_args: dict, key: str, expected: Any) -> bool:
    """key "name" -> equality (numbers within tolerance); key "name~" -> case-insensitive substring."""
    substring = key.endswith("~")
    name = key.rstrip("~")
    if name not in actual_args:
        return False
    actual = actual_args[name]
    if substring:
        return str(expected).lower().replace("-", "").replace(" ", "") in str(actual).lower().replace("-", "").replace(" ", "")
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        try:
            number = float(str(actual).lower().replace(",", "").replace("$", "").rstrip("k")) * (
                1_000 if str(actual).lower().endswith("k") else 1)
        except ValueError:
            return False
        return abs(number - expected) <= max(1.0, 0.03 * abs(expected))
    return _normalize(actual) == _normalize(expected)


@dataclass
class CaseScore:
    tool_recall: float
    tool_precision: float
    args_accuracy: float
    grounding: float
    language_match: bool
    passed: bool
    called_tools: list[str] = field(default_factory=list)
    missing_tools: list[str] = field(default_factory=list)
    failed_arg_checks: list[str] = field(default_factory=list)
    ungrounded_numbers: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def score_turn(case: dict, answer: str, tool_calls: list[dict], user_messages: list[str],
               reference_texts: tuple[str, ...] = ()) -> CaseScore:
    """tool_calls: [{"name": ..., "args": {...}, "output": "<json>"}, ...]

    reference_texts: texts the agent legitimately knows besides tool results, i.e. its system
    instructions and tool descriptions (e.g. "90% range").
    """
    expected = set(case.get("expected_tools", []))
    allowed = expected | set(case.get("allowed_tools", []))
    called = [call["name"] for call in tool_calls]
    called_set = set(called)

    tool_recall = len(expected & called_set) / len(expected) if expected else 1.0
    tool_precision = len(called_set & allowed) / len(called_set) if called_set else (1.0 if not expected else 0.0)

    checks, failed = 0, []
    for tool, arg_checks in case.get("expected_args", {}).items():
        calls = [call["args"] for call in tool_calls if call["name"] == tool]
        for key, value in arg_checks.items():
            checks += 1
            if not any(_check_arg(args, key, value) for args in calls):
                failed.append(f"{tool}.{key}={value}")
    # Forbidden arguments: e.g. include_suspicious=true when the user did not ask for it.
    for tool, arg_checks in case.get("forbidden_args", {}).items():
        calls = [call["args"] for call in tool_calls if call["name"] == tool]
        for key, value in arg_checks.items():
            checks += 1
            if any(_check_arg(args, key, value) for args in calls):
                failed.append(f"{tool}.{key}={value} (forbidden)")
    args_accuracy = (checks - len(failed)) / checks if checks else 1.0

    sources = [n for call in tool_calls for n in numbers_in_tool_text(call.get("output", ""))]
    sources += [n for call in tool_calls for n in _numbers_in_json(call.get("args", {}))]
    sources += [n for message in user_messages for n in extract_numbers(message)]
    sources += [n for text in reference_texts for n in extract_numbers(text)]
    sources += list(ALWAYS_GROUNDED)
    checked = [n for n in extract_numbers(URL.sub("", answer)) if abs(n) >= MIN_CHECKED_NUMBER]
    ungrounded = [n for n in checked if not any(_close(n, s) for s in sources)]
    grounding = 1 - len(ungrounded) / len(checked) if checked else 1.0

    expected_language = case.get("language") or detect_language(user_messages[-1])
    language_match = detect_language(answer) == expected_language

    passed = (tool_recall == 1.0 and args_accuracy == 1.0 and grounding >= GROUNDING_PASS
              and language_match and bool(answer.strip()))
    return CaseScore(
        tool_recall=round(tool_recall, 3), tool_precision=round(tool_precision, 3),
        args_accuracy=round(args_accuracy, 3), grounding=round(grounding, 3),
        language_match=language_match, passed=passed, called_tools=called,
        missing_tools=sorted(expected - called_set), failed_arg_checks=failed, ungrounded_numbers=ungrounded,
    )
