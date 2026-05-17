import pytest
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

def test_home_endpoint():
    response = client.get("/", headers={"X-API-Key": "your-secret-api-key"})
    assert response.status_code == 200
    assert response.json() == {"message": "Fraud Detection API is Active!"}

def test_home_endpoint_no_api_key():
    response = client.get("/")
    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid API Key"}

def test_predict_endpoint():
    # Mock transaction data
    transaction_data = {
        "Time": 1.0,
        "V1": -1.0,
        "V2": 2.0,
        "V3": -3.0,
        "V4": 4.0,
        "V5": -5.0,
        "V6": 6.0,
        "V7": -7.0,
        "V8": 8.0,
        "V9": -9.0,
        "V10": 10.0,
        "V11": -11.0,
        "V12": 12.0,
        "V13": -13.0,
        "V14": 14.0,
        "V15": -15.0,
        "V16": 16.0,
        "V17": -17.0,
        "V18": 18.0,
        "V19": -19.0,
        "V20": 20.0,
        "V21": -21.0,
        "V22": 22.0,
        "V23": -23.0,
        "V24": 24.0,
        "V25": -25.0,
        "V26": 26.0,
        "V27": -27.0,
        "V28": 28.0,
        "Amount": 100.0
    }
    response = client.post("/predict", json=transaction_data, headers={"X-API-Key": "your-secret-api-key"})
    assert response.status_code == 200
    assert "result" in response.json()
    assert "confidence" in response.json()

def test_predict_endpoint_no_api_key():
    transaction_data = {"Time": 1.0, "V1": -1.0, "V2": 2.0, "V3": -3.0, "V4": 4.0, "V5": -5.0, "V6": 6.0, "V7": -7.0, "V8": 8.0, "V9": -9.0, "V10": 10.0, "V11": -11.0, "V12": 12.0, "V13": -13.0, "V14": 14.0, "V15": -15.0, "V16": 16.0, "V17": -17.0, "V18": 18.0, "V19": -19.0, "V20": 20.0, "V21": -21.0, "V22": 22.0, "V23": -23.0, "V24": 24.0, "V25": -25.0, "V26": 26.0, "V27": -27.0, "V28": 28.0, "Amount": 100.0}
    response = client.post("/predict", json=transaction_data)
    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid API Key"}