import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.pipeline.predict_pipeline import PredictPipeline
from src.logger import logging

ml_artifacts = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Loads PredictPipeline (and therefore all three model artifacts) ONCE
    when the app starts, not per-request. See our earlier system design
    discussion — this is the "load once, serve many" principle in action.
    """
    logging.info("Starting up: loading PredictPipeline...")
    try:
        ml_artifacts["pipeline"] = PredictPipeline()
        logging.info("PredictPipeline loaded successfully")
    except Exception as e:
        logging.error(f"Failed to load PredictPipeline: {e}")
        ml_artifacts["pipeline"] = None
    yield
    logging.info("Shutting down")


app = FastAPI(
    title="Credit Card Customer Segmentation API",
    description="Predicts which behavioral segment a credit card customer belongs to.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CustomerFeatures(BaseModel):
    BALANCE: float = Field(..., ge=0)
    BALANCE_FREQUENCY: float = Field(..., ge=0, le=1)
    ONEOFF_PURCHASES: float = Field(..., ge=0)
    INSTALLMENTS_PURCHASES: float = Field(..., ge=0)
    CASH_ADVANCE: float = Field(..., ge=0)
    PURCHASES_FREQUENCY: float = Field(..., ge=0, le=1)
    ONEOFF_PURCHASES_FREQUENCY: float = Field(..., ge=0, le=1)
    CASH_ADVANCE_FREQUENCY: float = Field(..., ge=0)
    PURCHASES_TRX: int = Field(..., ge=0)
    CREDIT_LIMIT: float = Field(..., ge=0)
    PAYMENTS: float = Field(..., ge=0)
    MINIMUM_PAYMENTS: float = Field(..., ge=0)
    PRC_FULL_PAYMENT: float = Field(..., ge=0, le=1)
    TENURE: int = Field(..., ge=1, le=12)


class SegmentResponse(BaseModel):
    segment: int
    description: str
    segment_size_in_training_data: int


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": ml_artifacts.get("pipeline") is not None,
    }


@app.post("/predict", response_model=SegmentResponse)
def predict(customer: CustomerFeatures):
    pipeline = ml_artifacts.get("pipeline")
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run the training pipeline, then restart the API.",
        )

    result = pipeline.predict(customer.model_dump())
    return result


frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")