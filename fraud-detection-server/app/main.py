import signal
import time
from contextlib import asynccontextmanager
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow.entities.model_registry import ModelVersion
from sklearn.pipeline import Pipeline

from app.core.model_loader import MlflowModelLoader
from app.core.prediction_request import PredictionRequest

model: Optional[Pipeline] = None
model_version: Optional[ModelVersion] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to handle startup and shutdown events.
    """
    global model, model_version
    try:
        # Load the model during app startup
        load_model(max_retries=10, retry_delay=30)
        yield

    finally:
        # Clean up on shutdown
        model = None
        model_version = None
        app.state.model = None
        app.state.model_version = None
        print("Model resources have been cleaned up.")


app = FastAPI(title="Fraud Detection API", version="0.1", lifespan=lifespan)


@app.get("/", summary="Root endpoint", tags=["general"])
async def root():
    """
    Root endpoint of the Fraud Detection API.

    This endpoint can be used as a simple health check or to display basic API information.
    """
    return {
        "message": "Welcome to the Fraud Detection API",
        "version": app.version,
        "docs_url": "/docs",
        "openapi_url": "/openapi.json",
    }


@app.post(
    "/predict", summary="Predict Fraud", response_description="The prediction result"
)
async def predict_fraud(request: PredictionRequest):
    """
    Endpoint to predict whether a transaction is fraudulent.
    """
    try:
        start_time = time.time()  # Record the start time

        loaded_model: Optional[Pipeline] = app.state.model
        if loaded_model is None:
            raise HTTPException(status_code=500, detail="Model not loaded.")

        input_data = request.model_dump()
        input_df = pd.DataFrame([input_data])

        prediction_proba = loaded_model.predict_proba(input_df)

        fraud_probability = float(prediction_proba[0][1])

        prediction_time = time.time() - start_time

        return {
            "is_fraud": False if fraud_probability <= 0.5 else True,
            "threshold": 0.5,
            "probability": fraud_probability,
            "model_version": (app.state.model_version or None)
            and app.state.model_version.version,
            "prediction_time": prediction_time,
        }
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/reload", summary="Reload model", response_description="Reload model from MLFlow"
)
async def reload():
    """
    Endpoint to predict whether a transaction is fraudulent.
    """
    try:
        success = load_model()
        if not success:
            raise HTTPException(status_code=500, detail="Coud not reload the model.")
        return {
            "status": "success",
            "message": "Model reloaded successfully",
            "model_version": (app.state.model_version or None)
            and app.state.model_version.version,
        }
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def load_model(max_retries: int = 3, retry_delay: int = 5) -> bool:
    global model, model_version

    def signal_handler(signum, frame):
        print("\nInterrupt received, stopping model loading process...")
        raise KeyboardInterrupt

    # Set up the signal handler
    signal.signal(signal.SIGINT, signal_handler)

    retries = 0
    while max_retries == -1 or retries < max_retries:
        try:
            model, model_version = MlflowModelLoader.load_model_pipeline()
            if model is not None and model_version is not None:
                app.state.model = model
                app.state.model_version = model_version
                print(f"Loaded model version: {model_version.version}")
                return True
            else:
                retry_count = (
                    f"Attempt {retries + 1}"
                    if max_retries != -1
                    else f"Attempt {retries + 1} (retrying indefinitely)"
                )
                print(f"Model or model version is None. Retrying... ({retry_count})")
                retries += 1
                time.sleep(retry_delay)
        except Exception as e:
            print(f"Error loading model: {e}")
            retries += 1
            retry_count = (
                f"Attempt {retries}/{max_retries}"
                if max_retries != -1
                else f"Attempt {retries} (retrying indefinitely)"
            )
            print(f"Retrying in {retry_delay} seconds... ({retry_count})")
            time.sleep(retry_delay)

    if max_retries != -1:
        print(f"Failed to load model after {max_retries} attempts")
    return False
