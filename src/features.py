import numpy as np

def create_daily_dataset(df):

    logiciel_agg = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }

    df_day = df.resample('D').agg(logiciel_agg)

    print(df_day.head())
    print(f"Nouvelle taille : {df_day.shape}")

    return df_day


def add_log_features(df_day):

    df_day['Log_Price'] = np.log(df_day['Close'])
    df_day['Log_Return'] = df_day['Log_Price'].diff()

    return df_day