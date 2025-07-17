# scripts/configuration.py
conf = {
    # "LinearRegression": {
    #     "feature_encoding_methods": ["one-hot", "ordinal"],
    #     "model_params": {}
    # },
    "RandomForest": {
        "feature_encoding_methods": ["one-hot", "ordinal"],
        "model_params": {
            "n_estimators": 100,
            "random_state": 42,
            "n_jobs": 2
        }
    },
    "LGBM": {
        "feature_encoding_methods": ["one-hot", "ordinal"],
        "model_params": {
            "n_estimators": 100,
            "random_state": 42,
            "n_jobs": 2
        }
    },
    "XGBoost": {
        "feature_encoding_methods": ["one-hot", "ordinal"],
        "model_params": {
            "n_estimators": 100,
            "random_state": 42,
            "n_jobs": 2
        }
    }
}