from fastapi import FastAPI

app = FastAPI(title="Bitcoin Forecast API")


@app.get("/")
def home():

    return {
        "message": "Bitcoin Prediction API (ARIMA + XGBoost + LSTM)"
    }


@app.get("/predict")
def predict(days: int = 7):

    return {
        "forecast_days": days,
        "message": "Prediction endpoint working"
    }