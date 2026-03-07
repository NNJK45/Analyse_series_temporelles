import pandas as pd

def preprocess_data(df):

    # conversion timestamp -> date
    df['Date'] = pd.to_datetime(df['Timestamp'], unit='s')

    # tri chronologique
    df.sort_values('Date', inplace=True)

    # mettre Date comme index
    df.set_index('Date', inplace=True)

    # supprimer timestamp
    df.drop(columns=['Timestamp'], inplace=True)

    print(df.head())
    print(df.shape)
    print(df.isna().sum())

    return df