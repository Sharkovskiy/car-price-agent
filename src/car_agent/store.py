"""Loads the price model and the listings exported by the Kaggle notebook."""
from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd

REQUIRED_FILES = ("model.json", "meta.json", "car_price_predictor.py", "listings.parquet")


@dataclass
class DataStore:
    cpp: ModuleType          # car_price_predictor module shipped together with the model
    predictor: Any           # cpp.CarPricePredictor
    listings: pd.DataFrame   # cleaned listings with precomputed estimates and deal scores
    data_snapshot: str | None


def _import_predictor_module(path: Path) -> ModuleType:
    """Import car_price_predictor.py from the artifacts folder, so model and code always match."""
    spec = importlib.util.spec_from_file_location("car_price_predictor", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["car_price_predictor"] = module
    spec.loader.exec_module(module)
    return module


def load_store(artifacts_dir: Path) -> DataStore:
    missing = [name for name in REQUIRED_FILES if not (artifacts_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing artifacts in {artifacts_dir}: {', '.join(missing)}. See artifacts/README.md."
        )
    cpp = _import_predictor_module(artifacts_dir / "car_price_predictor.py")
    predictor = cpp.CarPricePredictor.load(artifacts_dir)
    listings = pd.read_parquet(artifacts_dir / "listings.parquet")
    return DataStore(cpp=cpp, predictor=predictor, listings=listings,
                     data_snapshot=predictor.metadata.get("data_snapshot"))
