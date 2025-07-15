conf = {
    "LinearRegression": {
        "feature_encoding_methods": ["one-hot", "label"],
        "model_params": {}
    },
    "RandomForest": {
        "feature_encoding_methods": ["one-hot", "label"],
        "model_params": {
            "n_estimators": 100,
            "random_state": 42,
            "n_jobs": 2
        }
    }
}