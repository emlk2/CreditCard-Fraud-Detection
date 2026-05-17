# routes.py - API endpoints for the fraud detection service

import os
import logging
from fastapi import APIRouter, Request, Depends, HTTPException, status
from slowapi import Limiter
from slowapi.util import get_remote_address

from src.models import TransactionData
from src.utils import get_model, predict_fraud

logger = logging.getLogger(__name__)

router = APIRouter()

limiter = Limiter(key_func=get_remote_address)

# API Key for authentication (should be in config, but for simplicity)
API_KEY = os.getenv("API_KEY", "your-secret-api-key")

def verify_api_key(request: Request):
    api_key = request.headers.get("X-API-Key")
    if api_key != API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API Key")
    return api_key

# Home endpoint
@router.get("/")
@limiter.limit("10/minute")
def home_page(request: Request, api_key: str = Depends(verify_api_key)):
    logger.info(f"Home endpoint accessed from {get_remote_address(request)}")
    return {"message": "Fraud Detection API is Active!"}

# Single prediction endpoint (async)
@router.post("/predict")
@limiter.limit("100/minute")
async def predict_fraud_endpoint(
    request: Request,
    transaction: TransactionData,
    api_key: str = Depends(verify_api_key)
):
    logger.info(f"Predict endpoint called from {get_remote_address(request)}")
    model = get_model()
    if model is None:
        logger.error("Model not loaded")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded")
    result = await predict_fraud(model, transaction.dict())
    logger.info(f"Prediction result: {result['result']}")
    return result