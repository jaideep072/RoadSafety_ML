# Artifacts Directory Guide

This directory stores serialized model binaries, fitted scalers, and encoder transformers.

## Structure

```text
artifacts/
├── models/        # Pickled estimator binaries (.pkl / .joblib)
│   ├── linear_reg.pkl
│   ├── linear_ridge.pkl
│   ├── linear_lasso.pkl
│   ├── linear_elasticnet.pkl
│   ├── logistic_reg.pkl
│   ├── logistic_ridge.pkl
│   ├── logistic_lasso.pkl
│   ├── logistic_elasticnet.pkl
│   ├── dt_classifier.pkl
│   ├── dt_regressor.pkl
│   ├── rf_classifier.pkl
│   ├── rf_regressor.pkl
│   ├── gb_classifier.pkl
│   ├── gb_regressor.pkl
│   ├── xgb_classifier.pkl
│   ├── xgb_regressor.pkl
│   ├── lgbm_classifier.pkl
│   ├── lgbm_regressor.pkl
│   ├── adaboost_classifier.pkl
│   └── adaboost_regressor.pkl
├── scalers/       # Fitted StandardScaler & MinMaxScaler objects
│   └── scaler.pkl
└── encoders/      # Fitted SimpleImputer & Categorical encoders
    └── imputer.pkl
```

## Regeneration

To train and re-serialize all models and transformers, run:

```bash
python scripts/train_models.py --force
```
