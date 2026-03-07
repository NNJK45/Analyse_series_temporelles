import pandas as pd

def load_data(path):
    df = pd.read_csv(path)

    print(df.head())
    print(df.shape)
    print(df.isna().sum())

    return df