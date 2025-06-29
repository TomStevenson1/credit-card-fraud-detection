from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Any, List, Optional, Union, cast

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler

# Make the project root importable
sys.path.append(str(Path(__file__).parent.parent))
from training.train import NeuralNetwork, LogisticRegressionModel  # noqa: E402


# ───────────────────────────────
# Pydantic models
# ───────────────────────────────
class TransactionData(BaseModel):
    """Schema for a single credit-card transaction."""
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float


class PredictionResponse(BaseModel):
    fraud_probability: float
    is_fraud: bool
    model_used: str
    confidence: str


# ───────────────────────────────
# Helper typing
# ───────────────────────────────
Scaler = StandardScaler
SklearnClassifier = Any                       # narrow if you wish
PyTorchModel = Union[nn.Module, SklearnClassifier]


# ───────────────────────────────
# Core detector
# ───────────────────────────────
class FraudDetector:
    """Wraps a scaler + ML model."""

    def __init__(self, model_path: str, scaler_path: str, model_type: str) -> None:
        self.model_type = model_type
        self.model: Optional[PyTorchModel] = None
        self.scaler: Optional[Scaler] = None

        # Load scaler
        with open(scaler_path, "rb") as f:
            self.scaler = cast(Scaler, pickle.load(f))

        # Load model
        if model_type in {"logreg_pytorch", "nn"}:
            self.model = (
                LogisticRegressionModel(29)
                if model_type == "logreg_pytorch"
                else NeuralNetwork(29)
            )
            self.model.load_state_dict(
                torch.load(model_path, map_location=torch.device("cpu"))
            )
            self.model.eval()
        else:  # sklearn
            with open(model_path, "rb") as f:
                self.model = cast(SklearnClassifier, pickle.load(f))

    # ----------------------------
    def predict(self, data: TransactionData) -> PredictionResponse:
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model or scaler not loaded")

        X_scaled = self.scaler.transform(
            pd.DataFrame([data.dict()]).drop(columns=["Time"])
        )

        # Compute probability
        if self.model_type in {"logreg_pytorch", "nn"}:
            tensor = torch.tensor(X_scaled, dtype=torch.float32)
            if isinstance(self.model, nn.Module):
                prob = self.model(tensor).item()
            else:
                raise RuntimeError("Expected PyTorch model")
        elif self.model_type == "sgd":
            if hasattr(self.model, 'decision_function'):
                model = cast(Any, self.model)
                dv = model.decision_function(X_scaled)[0]
                prob = 1 / (1 + np.exp(-dv))
            else:
                raise RuntimeError("Expected SGD classifier")
        else:  # sklearn logistic regression
            if hasattr(self.model, 'predict_proba'):
                model = cast(Any, self.model)
                prob = model.predict_proba(X_scaled)[0, 1]
            else:
                raise RuntimeError("Expected sklearn classifier")

        is_fraud = prob >= 0.5
        confidence = (
            "HIGH"
            if prob >= 0.8 or prob <= 0.2
            else "MEDIUM"
            if prob >= 0.6 or prob <= 0.4
            else "LOW"
        )

        return PredictionResponse(
            fraud_probability=float(prob),
            is_fraud=is_fraud,
            model_used=self.model_type,
            confidence=confidence,
        )


# ───────────────────────────────
# FastAPI plumbing
# ───────────────────────────────
app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Real-time credit-card fraud detection using ML models",
    version="1.0.0",
)

fraud_detector: Optional[FraudDetector] = None


@app.on_event("startup")
async def startup_event() -> None:
    global fraud_detector

    model_dir = Path("output/models")
    models = list(model_dir.glob("*_model.pkl"))
    if not models:
        raise RuntimeError("No trained models found in output/models/")

    model_path = models[0]
    model_type = model_path.stem.replace("_model", "")
    scaler_path = model_dir / f"{model_type}_scaler.pkl"
    if not scaler_path.exists():
        raise RuntimeError(f"Scaler not found: {scaler_path}")

    fraud_detector = FraudDetector(str(model_path), str(scaler_path), model_type)
    print(f"✅ Loaded {model_type} from {model_path}")


@app.get("/")
async def root():
    return {"message": "Credit Card Fraud Detection API", "status": "running"}


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": fraud_detector is not None,
        "model_type": fraud_detector.model_type if fraud_detector else None,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(transaction: TransactionData):
    if fraud_detector is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    try:
        return fraud_detector.predict(transaction)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}") from exc


@app.post("/predict_batch")
async def predict_batch(transactions: List[TransactionData]):
    if fraud_detector is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    try:
        preds = [fraud_detector.predict(t).dict() for t in transactions]
        return {"predictions": preds, "total_transactions": len(transactions)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {exc}") from exc


@app.get("/model_info")
async def model_info():
    if fraud_detector is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    return {
        "model_type": fraud_detector.model_type,
        "features": 29,  # V1–V28 + Amount
        "input_schema": TransactionData.schema(),
    }


if __name__ == "__main__":
    try:
        import uvicorn

        uvicorn.run(app, host="0.0.0.0", port=8000)
    except ImportError:
        print("Install uvicorn to run the server:  pip install uvicorn")
