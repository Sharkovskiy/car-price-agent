"""Agent tools: plain Python functions over the price model and the listings store.

The tools never raise on bad input from the LLM. Errors are returned as {"error": ...},
so that the model can correct its arguments or explain the problem to the user.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from car_agent.store import DataStore

US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "district of columbia": "dc",
    "florida": "fl", "georgia": "ga", "hawaii": "hi", "idaho": "id", "illinois": "il",
    "indiana": "in", "iowa": "ia", "kansas": "ks", "kentucky": "ky", "louisiana": "la",
    "maine": "me", "maryland": "md", "massachusetts": "ma", "michigan": "mi", "minnesota": "mn",
    "mississippi": "ms", "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv",
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny",
    "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok", "oregon": "or",
    "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc", "south dakota": "sd",
    "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt", "virginia": "va",
    "washington": "wa", "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy",
}

MANUFACTURER_ALIASES = {
    "chevy": "chevrolet", "vw": "volkswagen", "mercedes": "mercedes-benz", "benz": "mercedes-benz",
    "mercedes benz": "mercedes-benz", "alfa romeo": "alfa-romeo", "aston martin": "aston-martin",
    "harley davidson": "harley-davidson",
}

LISTING_FIELDS = [
    "id", "year", "manufacturer", "model", "odometer", "price", "predicted_price", "deal_pct",
    "deal_verdict", "condition", "type", "seller_type", "state", "region", "posting_date", "url",
]

SORT_OPTIONS = {
    "best_deal": ("deal_pct", True),
    "price_low_to_high": ("price", True),
    "price_high_to_low": ("price", False),
    "newest_model_year": ("year", False),
    "lowest_mileage": ("odometer", True),
    "most_recent_posting": ("posting_date", False),
}


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    return str(value)


def to_number(value: Any) -> float | None:
    """Accept 15000, "15000", "15,000" or "$15k"; return None when not a number."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().lower().replace(",", "").replace("$", "").replace(" ", "")
    multiplier = 1_000 if text.endswith("k") else 1
    try:
        return float(text.rstrip("k")) * multiplier
    except ValueError:
        return None


class CarTools:
    """The five agent tools, bound to one data store."""

    def __init__(self, store: DataStore, max_results: int = 10):
        self.store = store
        self.cpp = store.cpp
        self.predictor = store.predictor
        self.listings = store.listings
        self.max_results = max_results
        self.data_note = f"Craigslist asking prices, USA, {store.data_snapshot}; USD and miles."

    # ---- normalization helpers ------------------------------------------------
    def norm_make(self, value: Any) -> str | None:
        text = self.cpp.normalize_text(value)
        if not isinstance(text, str):
            return None
        return MANUFACTURER_ALIASES.get(text, text)

    def norm_state(self, value: Any) -> str | None:
        text = self.cpp.normalize_text(value)
        if not isinstance(text, str):
            return None
        return US_STATES.get(text, text)

    def family_of(self, manufacturer: Any, model: Any) -> str | None:
        """Model family key used in the store, e.g. ("Toyota", "RAV-4 XLE") -> "toyota rav4"."""
        make = self.norm_make(manufacturer)
        family = self.cpp.normalize_model_name(model, make)
        return f"{make} {family}" if make and isinstance(family, str) else None

    def records(self, frame: pd.DataFrame) -> list[dict]:
        """Compact JSON-friendly representation of listings."""
        out = frame[LISTING_FIELDS].copy()
        out["posting_date"] = pd.to_datetime(out["posting_date"]).dt.strftime("%Y-%m-%d")
        out = out.astype(object).where(out.notna(), None)
        return out.to_dict("records")

    def top_families(self, make: str, n: int = 25) -> list[str]:
        """Most common model families of a manufacturer, without the manufacturer prefix."""
        counts = self.listings.loc[self.listings["manufacturer"] == make, "family"].value_counts().head(n)
        return [name[len(make) + 1:] for name in counts.index]

    def _limit(self, value: Any, default: int) -> int:
        return int(max(1, min(to_number(value) or default, self.max_results)))

    # ---- tools ------------------------------------------------------------------
    def estimate_price(self, year=None, odometer=None, manufacturer=None, model=None, condition=None,
                       cylinders=None, fuel=None, title_status=None, transmission=None, drive=None,
                       size=None, body_type=None, paint_color=None, state=None) -> dict:
        """Model estimate with a 90% range for a car described by the user."""
        car = {
            "year": to_number(year), "odometer": to_number(odometer), "manufacturer": self.norm_make(manufacturer),
            "model": model, "condition": condition, "cylinders": cylinders, "fuel": fuel,
            "title_status": title_status, "transmission": transmission, "drive": drive, "size": size,
            "type": body_type, "paint_color": paint_color, "state": self.norm_state(state),
        }
        car = {key: value for key, value in car.items() if not self.cpp.is_missing(value)}
        result = self.predictor.predict_one(car, top_k=5)
        keep = ["price", "price_low", "price_high", "currency", "reliable", "top_factors", "assumed_values", "warnings"]
        return {**{key: result[key] for key in keep}, "data_note": self.data_note}

    def search_listings(self, manufacturer=None, model=None, year_min=None, year_max=None, price_min=None,
                        price_max=None, odometer_max=None, state=None, body_type=None, fuel=None,
                        transmission=None, drive=None, condition=None, seller_type=None,
                        sort_by="best_deal", limit=5, include_suspicious=False) -> dict:
        """Filter real listings and return the top matches with their deal scores."""
        listings = self.listings
        mask = pd.Series(True, index=listings.index)
        applied: dict[str, Any] = {}

        make = self.norm_make(manufacturer)
        if make:
            mask &= listings["manufacturer"] == make
            applied["manufacturer"] = make
        if model:
            if not make:
                return {"error": "'model' requires 'manufacturer'."}
            family = self.family_of(make, model)
            mask &= listings["family"] == family
            applied["model_family"] = family

        numeric_filters = [
            ("year", year_min, "ge"), ("year", year_max, "le"), ("price", price_min, "ge"),
            ("price", price_max, "le"), ("odometer", odometer_max, "le"),
        ]
        for column, raw_value, op in numeric_filters:
            value = to_number(raw_value)
            if value is not None:
                mask &= listings[column].ge(value) if op == "ge" else listings[column].le(value)
                applied[f"{column}_{'min' if op == 'ge' else 'max'}"] = value

        text_filters = {
            "state": self.norm_state(state), "type": self.cpp.normalize_text(body_type),
            "fuel": self.cpp.normalize_text(fuel), "transmission": self.cpp.normalize_text(transmission),
            "drive": self.cpp.normalize_text(drive), "condition": self.cpp.normalize_text(condition),
            "seller_type": self.cpp.normalize_text(seller_type),
        }
        for column, value in text_filters.items():
            if isinstance(value, str):
                mask &= listings[column] == value
                applied[column] = value

        if not include_suspicious:
            mask &= listings["deal_verdict"] != "suspiciously_cheap"

        found = listings[mask]
        sort_by = sort_by if sort_by in SORT_OPTIONS else "best_deal"
        column, ascending = SORT_OPTIONS[sort_by]
        result = {
            "total_found": int(len(found)),
            "filters_applied": applied,
            "sort_by": sort_by,
            "results": self.records(found.sort_values(column, ascending=ascending).head(self._limit(limit, 5))),
        }
        if found.empty and make:
            result["hint"] = f"Known model families for {make}: {', '.join(self.top_families(make))}"
        return result

    def find_comparables(self, manufacturer, model, year, odometer=None, state=None, limit=8,
                         exclude_id=None) -> dict:
        """Most similar real listings: same model family, close year and mileage, preferably the same state."""
        make = self.norm_make(manufacturer)
        family = self.family_of(make, model)
        year = to_number(year)
        odometer = to_number(odometer)
        state = self.norm_state(state)
        if year is None:
            return {"error": "'year' must be a number."}

        pool = self.listings[self.listings["family"] == family]
        if exclude_id is not None and to_number(exclude_id) is not None:
            pool = pool[pool["id"] != int(to_number(exclude_id))]
        if pool.empty:
            hint = f" Known model families for {make}: {', '.join(self.top_families(make))}" if make else ""
            return {"error": f"No listings for model family '{family}'.{hint}"}

        # Widen the window step by step until at least 3 comparables are found.
        for year_tolerance, odometer_tolerance in [(1, 25_000), (2, 50_000), (3, None)]:
            candidates = pool[(pool["year"] - year).abs() <= year_tolerance]
            if odometer is not None and odometer_tolerance is not None:
                candidates = candidates[(candidates["odometer"] - odometer).abs() <= odometer_tolerance]
            if len(candidates) >= 3:
                break
        if candidates.empty:
            return {"model_family": family, "count": 0, "results": []}

        distance = (candidates["year"] - year).abs()
        if odometer is not None:
            distance = distance + (candidates["odometer"] - odometer).abs() / 25_000
        if state:
            distance = distance + 0.5 * (candidates["state"] != state)
        closest = candidates.assign(_distance=distance).nsmallest(self._limit(limit, 8), "_distance")

        def price_stats(frame: pd.DataFrame) -> dict:
            return {
                "median": float(frame["price"].median()),
                "p25": float(frame["price"].quantile(0.25)),
                "p75": float(frame["price"].quantile(0.75)),
            }

        result: dict[str, Any] = {
            "model_family": family,
            "search_window": {"year_tolerance": year_tolerance, "odometer_tolerance": odometer_tolerance},
            "count": int(len(candidates)),
            # Explicit scope, so that national numbers are not presented as local ones.
            "price_stats_scope": "all states",
            "price_stats": price_stats(candidates),
        }
        if state:
            same_state = candidates[candidates["state"] == state]
            result["same_state_count"] = int(len(same_state))
            if len(same_state) >= 5:
                result["same_state_price_stats"] = price_stats(same_state)
        result["results"] = self.records(closest)
        result["data_note"] = self.data_note
        return result

    def evaluate_listing(self, listing_id) -> dict:
        """Compare the asking price of one listing with the model estimate and explain the difference."""
        number = to_number(listing_id)
        match = self.listings[self.listings["id"] == int(number)] if number is not None else self.listings.iloc[0:0]
        if match.empty:
            return {"error": f"Listing {listing_id} not found."}
        row = match.iloc[0]
        car = {field: row[field] for field in self.cpp.RAW_FIELDS}
        # Real listings: missing fields mean the same as in the training data, so no imputation.
        estimate = self.predictor.predict_one(car, top_k=4, impute_missing=False)
        return {
            "listing": self.records(match)[0],
            "estimate": {key: estimate[key] for key in ("price", "price_low", "price_high", "reliable")},
            "deal_pct": float(row["deal_pct"]),
            "deal_verdict": row["deal_verdict"],
            "top_factors": estimate["top_factors"],
            "warnings": estimate["warnings"],
            "data_note": self.data_note,
        }

    def list_known_values(self, field, manufacturer=None) -> dict:
        """Valid values of a field; for 'model' the most common model families of a manufacturer."""
        field = self.cpp.normalize_text(field)
        if field == "body_type":
            field = "type"
        if field == "model":
            make = self.norm_make(manufacturer)
            if not make:
                return {"error": "Provide 'manufacturer' to list its models."}
            return {"field": "model", "manufacturer": make, "values": self.top_families(make, n=40)}
        if field in self.cpp.TEXT_FIELDS:
            return {"field": field, "values": self.predictor.allowed_values(field)}
        return {"error": f"Unknown field '{field}'. Use one of: model, {', '.join(self.cpp.TEXT_FIELDS)}."}

    # ---- dispatch ---------------------------------------------------------------
    @property
    def functions(self) -> dict:
        return {
            "estimate_price": self.estimate_price,
            "search_listings": self.search_listings,
            "find_comparables": self.find_comparables,
            "evaluate_listing": self.evaluate_listing,
            "list_known_values": self.list_known_values,
        }

    def run(self, name: str, args: dict | None, max_chars: int = 6_000) -> str:
        """Execute a tool by name and return its JSON result (errors are returned, not raised)."""
        func = self.functions.get(name)
        if func is None:
            result: Any = {"error": f"Unknown tool '{name}'. Available: {', '.join(self.functions)}"}
        else:
            try:
                result = func(**(args or {}))
            except TypeError as error:
                result = {"error": f"Invalid arguments: {error}"}
            except Exception as error:  # the agent should see the problem instead of crashing
                result = {"error": f"{type(error).__name__}: {error}"}
        text = json.dumps(result, ensure_ascii=False, default=json_default)
        if len(text) > max_chars:
            text = text[:max_chars] + " ...[truncated]"
        return text
