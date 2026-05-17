# This file is the core of our server.
# Jupyter Notebook (.ipynb) is for analysis; real applications run in .py files.

# Standard library imports
import os
import asyncio
import logging

# Third-party imports
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

# Local imports
from src.utils import load_model
from src.routes import router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 1. Start the API Application
app = FastAPI(title="Credit Card Fraud Detection")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# Global model variable
model = None

# Load the model on startup
@app.on_event("startup")
def startup_event():
    global model
    model = load_model()
    logger.info("Model loaded successfully")

# Include routes
app.include_router(router)