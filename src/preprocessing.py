import pandas as pd


def preprocess_data(df):
    processed = df.copy()

    processed["Date"] = pd.to_datetime(processed["Timestamp"], unit="s")
    processed.sort_values("Date", inplace=True)
    processed.set_index("Date", inplace=True)
    processed.drop(columns=["Timestamp"], inplace=True)

    return processed
