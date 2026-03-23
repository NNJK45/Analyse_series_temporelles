from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

from src.data_loader import load_data
from src.features import add_log_features, create_daily_dataset
from src.preprocessing import preprocess_data


ModelName = Literal["xgboost", "arima", "hybrid"]
ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = ROOT_DIR / "data" / "btcusd_1-min_data.csv"
XGBOOST_FEATURES = ["t-1", "t-2", "SMA_7", "Std_7", "Rolling_Delta", "DayOfWeek"]


@lru_cache(maxsize=2)
def load_daily_features(data_path: str, file_mtime: float) -> pd.DataFrame:
    del file_mtime
    raw_df = load_data(data_path)
    minute_df = preprocess_data(raw_df)
    daily_df = create_daily_dataset(minute_df)
    return add_log_features(daily_df).dropna()


def build_ml_frame(df_day: pd.DataFrame) -> pd.DataFrame:
    df_ml = df_day.copy()
    df_ml["Delta"] = df_ml["Close"].diff()
    df_ml["t-1"] = df_ml["Close"].shift(1)
    df_ml["t-2"] = df_ml["Close"].shift(2)
    df_ml["SMA_7"] = df_ml["Close"].rolling(window=7).mean()
    df_ml["Std_7"] = df_ml["Close"].rolling(window=7).std()
    df_ml["Rolling_Delta"] = df_ml["Delta"].rolling(window=3).mean()
    df_ml["DayOfWeek"] = df_ml.index.dayofweek
    return df_ml.dropna().copy()


def train_xgboost_model(df_day: pd.DataFrame) -> tuple[xgb.XGBRegressor, float, pd.DataFrame]:
    df_ml = build_ml_frame(df_day)

    split = max(int(len(df_ml) * 0.8), 1)
    X_train = df_ml[XGBOOST_FEATURES].iloc[:split]
    X_test = df_ml[XGBOOST_FEATURES].iloc[split:]
    y_train = df_ml["Delta"].iloc[:split]

    model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        objective="reg:squarederror",
        random_state=42,
    )
    model.fit(X_train, y_train)

    rmse = 0.0
    if not X_test.empty:
        preds_delta = model.predict(X_test)
        preds_price = X_test["t-1"].to_numpy() + preds_delta
        rmse = float(np.sqrt(mean_squared_error(df_ml["Close"].iloc[split:], preds_price)))

    return model, rmse, df_ml


def forecast_xgboost(df_day: pd.DataFrame, days: int) -> dict:
    model, rmse, df_ml = train_xgboost_model(df_day)
    history = df_ml["Close"].tolist()
    deltas = df_ml["Delta"].tolist()
    next_date = df_ml.index[-1] + pd.Timedelta(days=1)

    predictions = []
    for _ in range(days):
        recent_closes = history[-7:]
        recent_deltas = deltas[-3:]
        frame = pd.DataFrame(
            [
                {
                    "t-1": history[-1],
                    "t-2": history[-2],
                    "SMA_7": float(np.mean(recent_closes)),
                    "Std_7": float(np.std(recent_closes, ddof=1)),
                    "Rolling_Delta": float(np.mean(recent_deltas)),
                    "DayOfWeek": next_date.dayofweek,
                }
            ]
        )
        predicted_delta = float(model.predict(frame[XGBOOST_FEATURES])[0])
        predicted_close = float(history[-1] + predicted_delta)

        predictions.append({"date": next_date.date().isoformat(), "close": round(predicted_close, 2)})
        history.append(predicted_close)
        deltas.append(predicted_delta)
        next_date += pd.Timedelta(days=1)

    return {"predictions": predictions, "rmse": round(rmse, 4)}


def forecast_arima(df_day: pd.DataFrame, days: int) -> dict:
    train_size = max(int(len(df_day) * 0.8), 1)
    train_log = df_day["Log_Price"].iloc[:train_size]
    test_log = df_day["Log_Price"].iloc[train_size:]

    model_fit = ARIMA(train_log, order=(5, 1, 0)).fit()
    rmse = 0.0
    if not test_log.empty:
        test_actual = df_day["Close"].iloc[train_size:]
        preds_log = model_fit.forecast(steps=len(test_log))
        preds_price = np.exp(preds_log)
        rmse = float(np.sqrt(mean_squared_error(test_actual, preds_price)))

    forecast_index = pd.date_range(df_day.index[-1] + pd.Timedelta(days=1), periods=days, freq="D")
    forecast_log = model_fit.forecast(steps=days)
    forecast_price = np.exp(forecast_log)
    predictions = [
        {"date": date.date().isoformat(), "close": round(float(price), 2)}
        for date, price in zip(forecast_index, forecast_price)
    ]

    return {"predictions": predictions, "rmse": round(rmse, 4)}


@lru_cache(maxsize=4)
def cached_forecast(model_name: ModelName, days: int, data_path: str, file_mtime: float) -> dict:
    df_day = load_daily_features(data_path, file_mtime)

    if len(df_day) < 40:
        raise ValueError("Not enough daily history to build a reliable forecast.")

    latest_close = round(float(df_day["Close"].iloc[-1]), 2)
    latest_date = df_day.index[-1].date().isoformat()

    if model_name == "xgboost":
        model_result = forecast_xgboost(df_day, days)
        return {
            "model": "xgboost",
            "latest_observation_date": latest_date,
            "latest_close": latest_close,
            **model_result,
        }

    if model_name == "arima":
        model_result = forecast_arima(df_day, days)
        return {
            "model": "arima",
            "latest_observation_date": latest_date,
            "latest_close": latest_close,
            **model_result,
        }

    xgb_result = forecast_xgboost(df_day, days)
    arima_result = forecast_arima(df_day, days)
    predictions = []
    for xgb_pred, arima_pred in zip(xgb_result["predictions"], arima_result["predictions"]):
        close = round((xgb_pred["close"] + arima_pred["close"]) / 2, 2)
        predictions.append({"date": xgb_pred["date"], "close": close})

    return {
        "model": "hybrid",
        "latest_observation_date": latest_date,
        "latest_close": latest_close,
        "predictions": predictions,
        "rmse": {
            "xgboost": xgb_result["rmse"],
            "arima": arima_result["rmse"],
        },
    }


def generate_forecast(
    days: int,
    model_name: ModelName = "hybrid",
    data_path: Path | str = DEFAULT_DATA_PATH,
) -> dict:
    if days < 1 or days > 30:
        raise ValueError("days must be between 1 and 30.")

    csv_path = Path(data_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    return cached_forecast(model_name, days, str(csv_path), csv_path.stat().st_mtime)
