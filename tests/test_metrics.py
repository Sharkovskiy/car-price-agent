import json

from car_agent.evaluation.metrics import detect_language, extract_numbers, score_turn


def test_extract_numbers_handles_separators_decimals_and_thousands():
    text = "Оцінка 18 285 USD (діапазон 13 405 - 24,677), похибка 10,7 %, пробіг 70 тис. миль, ~$18.3k"
    assert extract_numbers(text) == [18285, 13405, 24677, 10.7, 70000, 18300]


def test_detect_language():
    assert detect_language("Оцінка вартості: 18 285 USD для Toyota RAV4") == "uk"
    assert detect_language("Estimated price is 18,285 USD for a Toyota RAV4") == "en"
    assert detect_language("[link](https://example.com/very/long/english/url) Ціна нормальна") == "uk"


CASE = {
    "language": "en",
    "expected_tools": ["estimate_price"],
    "allowed_tools": ["find_comparables"],
    "expected_args": {"estimate_price": {"manufacturer": "toyota", "model~": "rav4", "year": 2016,
                                         "odometer": 80000, "state": "ca"}},
}
OUTPUT = json.dumps({"price": 18285, "price_low": 13405, "price_high": 24677, "reliable": True})
USER = ["How much is a 2016 Toyota RAV4 with 80k miles in California?"]


def test_perfect_turn_passes():
    calls = [{"name": "estimate_price", "output": OUTPUT,
              "args": {"manufacturer": "Toyota", "model": "RAV-4", "year": 2016, "odometer": "80k", "state": "California"}}]
    answer = "About $18,285 (range $13,405 - $24,677) for a 2016 RAV4 with 80,000 miles."
    score = score_turn(CASE, answer, calls, USER)
    assert score.passed, score
    assert score.tool_recall == score.tool_precision == score.args_accuracy == score.grounding == 1.0


def test_invented_number_and_wrong_args_are_detected():
    calls = [{"name": "estimate_price", "output": OUTPUT,
              "args": {"manufacturer": "toyota", "model": "rav4", "year": 2016, "odometer": 50000, "state": "ca"}},
             {"name": "search_listings", "output": "{}", "args": {}}]
    answer = "About $18,285, and similar cars sell for $21,900."
    score = score_turn(CASE, answer, calls, USER)
    assert not score.passed
    assert score.failed_arg_checks == ["estimate_price.odometer=80000"]
    assert score.ungrounded_numbers == [21900]
    assert score.tool_precision == 0.5


def test_typographic_minus_and_instruction_numbers_are_grounded():
    calls = [{"name": "estimate_price", "output": json.dumps({"price": 18285, "deal_pct": -34.6}),
              "args": {"manufacturer": "toyota", "model": "rav4", "year": 2016, "odometer": 80000, "state": "ca"}}]
    answer = "About $18,285 (90 % range), deal ‑35 %, i.e. (price - estimate) / estimate × 100."
    without_references = score_turn(CASE, answer, calls, USER)
    assert without_references.ungrounded_numbers == [90]
    with_references = score_turn(CASE, answer, calls, USER, reference_texts=("Estimate the price (with a 90% range).",))
    assert with_references.ungrounded_numbers == []
    assert with_references.passed


def test_forbidden_argument_fails_the_case():
    case = {"language": "en", "expected_tools": ["search_listings"],
            "forbidden_args": {"search_listings": {"include_suspicious": True}}}
    good = [{"name": "search_listings", "args": {"price_max": 15000}, "output": "{}"}]
    bad = [{"name": "search_listings", "args": {"price_max": 15000, "include_suspicious": True}, "output": "{}"}]
    assert score_turn(case, "Here are the listings.", good, ["Find SUVs"]).passed
    score = score_turn(case, "Here are the listings.", bad, ["Find SUVs"])
    assert not score.passed
    assert score.failed_arg_checks == ["search_listings.include_suspicious=True (forbidden)"]


def test_numbers_in_json_keys_are_sources():
    output = json.dumps({"price_stats": {"median": 27995, "p25": 25074.5, "p75": 30247.5}})
    calls = [{"name": "find_comparables", "args": {}, "output": output}]
    case = {"language": "en", "expected_tools": ["find_comparables"]}
    answer = "Median $27,995; 25th percentile $25,075; 75th percentile $30,250."
    assert score_turn(case, answer, calls, ["Similar listings?"]).ungrounded_numbers == []


def test_missing_tool_and_wrong_language():
    score = score_turn(CASE, "Ціна близько 18 000 доларів.", [], USER)
    assert score.tool_recall == 0.0
    assert score.missing_tools == ["estimate_price"]
    assert not score.language_match
    assert score.ungrounded_numbers == [18000]
