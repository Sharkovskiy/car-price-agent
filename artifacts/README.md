# Artifacts

This folder holds the model and the listings exported by the Kaggle notebook
`car-price-prediction`. Its contents are not versioned (see `.gitignore`).

Expected layout:

```
artifacts/
└── car_price_model/
    ├── model.json              # XGBoost booster
    ├── meta.json               # categories, vocabulary, price range calibration, metrics
    ├── car_price_predictor.py  # feature pipeline and predictor class
    └── listings.parquet        # cleaned listings with precomputed estimates
```

How to get it:

1. Run the Kaggle notebook (Save & Run All).
2. Open the notebook output (*Output* tab) and download the `car_price_model` folder.
3. Unpack it here, so that `artifacts/car_price_model/meta.json` exists.

Another location can be set with the `ARTIFACTS_DIR` environment variable.
