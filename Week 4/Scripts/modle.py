# scripts/model.py
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

MODEL_MAPPING = {
    'LinearRegression': LinearRegression,
    'RandomForest': RandomForestRegressor
}

def build_model(model_name, **kwargs):
    if model_name not in MODEL_MAPPING:
        raise ValueError(f"Model '{model_name}' not found. Available models: {list(MODEL_MAPPING.keys())}")
    model_class = MODEL_MAPPING[model_name]
    model_instance = model_class(**kwargs)
    return model_instance