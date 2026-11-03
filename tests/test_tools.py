"""Tool tests on the real artifacts (skipped when artifacts/car_price_model is not present)."""
import json

import pytest

from car_agent.config import load_settings
from car_agent.store import REQUIRED_FILES, load_store
from car_agent.tools import CarTools, to_number

settings = load_settings()
pytestmark = pytest.mark.skipif(
    not all((settings.artifacts_dir / name).exists() for name in REQUIRED_FILES),
    reason=f"artifacts not found in {settings.artifacts_dir}",
)


@pytest.fixture(scope="module")
def tools():
    return CarTools(load_store(settings.artifacts_dir))


def test_to_number():
    assert to_number("15k") == 15000
    assert to_number("$15,000") == 15000
    assert to_number(2016) == 2016
    assert to_number("abc") is None
    assert to_number(True) is None


def test_estimate_price_normalizes_user_input(tools):
    result = tools.estimate_price(year=2016, odometer="80k", manufacturer="Toyota", model="RAV-4 XLE", state="California")
    assert result["price_low"] < result["price"] < result["price_high"]
    assert result["reliable"] is True
    assert result["assumed_values"]  # secondary fields were filled with typical values


def test_estimate_price_flags_unknown_car(tools):
    result = tools.estimate_price(year=2024, odometer=3000, manufacturer="zaporozhets", model="968m")
    assert result["reliable"] is False
    assert any("zaporozhets" in warning for warning in result["warnings"])


def test_search_listings_filters_and_aliases(tools):
    result = tools.search_listings(manufacturer="Chevy", price_max="25k", state="Texas", limit=50)
    assert result["filters_applied"] == {"manufacturer": "chevrolet", "price_max": 25000.0, "state": "tx"}
    assert len(result["results"]) <= tools.max_results
    for row in result["results"]:
        assert row["manufacturer"] == "chevrolet" and row["price"] <= 25000 and row["state"] == "tx"
        assert row["deal_verdict"] != "suspiciously_cheap"
        assert row["seller_type"] in ("owner", "dealer", None)


def test_search_listings_unknown_model_returns_hint(tools):
    result = tools.search_listings(manufacturer="toyota", model="cybertruck")
    assert result["total_found"] == 0
    assert "Known model families for toyota" in result["hint"]


def test_find_comparables_reports_geographic_scope(tools):
    family = tools.listings["family"].value_counts().index[0]
    make = tools.listings.loc[tools.listings["family"] == family, "manufacturer"].iloc[0]
    row = tools.listings[tools.listings["family"] == family].iloc[0]
    result = tools.find_comparables(make, family[len(make) + 1:], int(row["year"]), row["odometer"], row["state"])
    assert result["price_stats_scope"] == "all states"
    assert "same_state_count" in result
    assert result["count"] >= len(result["results"]) > 0


def test_evaluate_listing(tools):
    listing_id = int(tools.listings["id"].iloc[0])
    result = tools.evaluate_listing(listing_id)
    assert result["listing"]["id"] == listing_id
    assert result["deal_verdict"]
    assert tools.evaluate_listing(-1) == {"error": "Listing -1 not found."}


def test_run_never_raises(tools):
    assert "Unknown tool" in json.loads(tools.run("no_such_tool", {}))["error"]
    assert "Invalid arguments" in json.loads(tools.run("evaluate_listing", {"wrong": 1}))["error"]
    assert tools.run("list_known_values", {"field": "model", "manufacturer": "vw"})
