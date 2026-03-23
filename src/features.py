import numpy as np


def create_daily_dataset(df):
    aggregation = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }

    df_day = df.resample("D").agg(aggregation)
    return df_day.dropna(subset=["Close"])


def add_log_features(df_day):
    features = df_day.copy()
    features["Log_Price"] = np.log(features["Close"])
    features["Log_Return"] = features["Log_Price"].diff()
    return features
