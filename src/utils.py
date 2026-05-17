import os
import joblib
import pandas as pd
import math
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

logger = logging.getLogger(__name__)

model: Any = None


def load_model() -> Any:
    """Load the machine learning model from the specified path."""
    global model
    model_path = os.getenv("MODEL_PATH", "random_forest_model.pkl")
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        raise RuntimeError(f"Model file not found at {model_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading model: {str(e)}")


def get_model() -> Any:
    """Return the loaded model instance."""
    return model


def sanitize_input(data: dict) -> dict:
    """Sanitize input data: check for NaN values and handle edge cases."""
    sanitized = {}
    for key, value in data.items():
        if isinstance(value, float) and math.isnan(value):
            raise ValueError(f"Invalid input: {key} cannot be NaN")
        sanitized[key] = value
    return sanitized


def predict_single(model: Any, data: dict) -> dict:
    """Predict fraud for a single transaction."""
    sanitized_data = sanitize_input(data)
    input_df = pd.DataFrame([sanitized_data])
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    result = "Fraud Suspected!" if prediction == 1 else "Normal Transaction"
    logger.info(f"Single prediction: {result} with score {probability:.4f}")
    return {
        "result": result,
        "fraud_score": float(probability),
        "prediction_code": int(prediction)
    }


async def predict_fraud(model: Any, data: dict) -> dict:
    """Asynchronously predict fraud using ThreadPoolExecutor."""
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor() as executor:
        result = await loop.run_in_executor(executor, predict_single, model, data)
    return result