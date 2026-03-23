from pathlib import Path

import pandas as pd


DEFAULT_COLUMNS = ["Timestamp", "Open", "High", "Low", "Close", "Volume"]


def load_data(path, nrows=None, columns=None):
    csv_path = Path(path)
    selected_columns = columns or DEFAULT_COLUMNS

    return pd.read_csv(csv_path, usecols=selected_columns, nrows=nrows)
