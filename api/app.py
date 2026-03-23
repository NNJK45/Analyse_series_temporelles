from typing import Literal

from fastapi import FastAPI, HTTPException, Query

from src.predictor import DEFAULT_DATA_PATH, generate_forecast

app = FastAPI(title="Bitcoin Forecast API", version="1.0.0")


@app.get("/")
def home():
    return {
        "message": "Bitcoin Prediction API",
        "available_models": ["xgboost", "arima", "hybrid"],
        "default_dataset": str(DEFAULT_DATA_PATH),
        "docs": "/docs",
    }


@app.get("/health")
def healthcheck():
    return {"status": "ok"}


@app.get("/predict")
def predict(
    days: int = Query(default=7, ge=1, le=30),
    model: Literal["xgboost", "arima", "hybrid"] = Query(default="hybrid"),
):
    try:
        forecast = generate_forecast(days=days, model_name=model)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    return {
        "forecast_days": days,
        **forecast,
    }
