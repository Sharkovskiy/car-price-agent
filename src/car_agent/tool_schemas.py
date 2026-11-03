"""JSON schemas of the agent tools in the OpenAI function-calling format (as seen by the LLM).

Descriptions are short on purpose: they are sent with every request.
"""
from car_agent.tools import SORT_OPTIONS


def _fn(name: str, description: str, properties: dict, required: list | None = None) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required or []},
        },
    }


S = {"type": "string"}
N = {"type": "number"}

CAR_PROPERTIES = {
    "year": {**N, "description": "Model year"},
    "odometer": {**N, "description": "Mileage in miles"},
    "manufacturer": {**S, "description": "Make, e.g. toyota, ford, bmw"},
    "model": {**S, "description": "Model, e.g. rav4, f-150, 3 series"},
    "condition": {**S, "description": "new, like new, excellent, good, fair, salvage"},
    "cylinders": {**N},
    "fuel": {**S, "description": "gas, diesel, hybrid, electric, other"},
    "title_status": {**S, "description": "clean, rebuilt, salvage, lien, missing, parts only"},
    "transmission": {**S, "description": "automatic, manual, other"},
    "drive": {**S, "description": "fwd, rwd, 4wd"},
    "size": {**S, "description": "compact, mid-size, full-size, sub-compact"},
    "body_type": {**S, "description": "sedan, suv, pickup, truck, coupe, hatchback, wagon, van, minivan, convertible, ..."},
    "paint_color": {**S},
    "state": {**S, "description": "US state, two-letter code or full name"},
}


def tool_schemas(max_results: int = 10) -> list[dict]:
    return [
        _fn(
            "estimate_price",
            "Estimate the market price (with a 90% range) of a car described by the user.",
            CAR_PROPERTIES,
            ["year", "manufacturer", "model"],
        ),
        _fn(
            "search_listings",
            "Search real used-car listings by filters. Returns the total count and the top matches with "
            "asking price, model estimate and deal score.",
            {
                "manufacturer": S, "model": S,
                "year_min": N, "year_max": N, "price_min": N, "price_max": N,
                "odometer_max": {**N, "description": "Maximum mileage in miles"},
                "state": CAR_PROPERTIES["state"], "body_type": CAR_PROPERTIES["body_type"],
                "fuel": S, "transmission": S, "drive": S, "condition": S,
                "seller_type": {**S, "enum": ["owner", "dealer"]},
                "sort_by": {**S, "enum": list(SORT_OPTIONS)},
                "limit": {**N, "description": f"Number of listings to return (max {max_results})"},
                "include_suspicious": {
                    "type": "boolean",
                    "description": "Default false. Set true ONLY if the user explicitly asks for listings priced below "
                                   "the typical range (possible damage, scams or down-payment ads).",
                },
            },
        ),
        _fn(
            "find_comparables",
            "Find real listings most similar to a given car (same model, close year and mileage) and their price statistics.",
            {
                "manufacturer": S, "model": S, "year": N,
                "odometer": {**N, "description": "Mileage in miles"},
                "state": CAR_PROPERTIES["state"],
                "limit": N,
                "exclude_id": {**N, "description": "Listing id to exclude (the listing being evaluated)"},
            },
            ["manufacturer", "model", "year"],
        ),
        _fn(
            "evaluate_listing",
            "Compare the asking price of a specific listing (by id) with the model estimate.",
            {"listing_id": N},
            ["listing_id"],
        ),
        _fn(
            "list_known_values",
            "List valid values of a field (e.g. model families of a manufacturer, body types, states). "
            "Use it when unsure how a model or value is spelled in the data.",
            {
                "field": {**S, "description": "model, manufacturer, body_type, fuel, condition, drive, state, ..."},
                "manufacturer": {**S, "description": "Required when field is 'model'"},
            },
            ["field"],
        ),
    ]
