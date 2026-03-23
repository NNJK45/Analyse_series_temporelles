import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error


def train_xgboost(df_day):

    df_ml = df_day.copy()

    df_ml['Delta'] = df_ml['Close'].diff()

    df_ml['t-1'] = df_ml['Close'].shift(1)
    df_ml['t-2'] = df_ml['Close'].shift(2)

    df_ml['SMA_7'] = df_ml['Close'].rolling(window=7).mean()
    df_ml['Std_7'] = df_ml['Close'].rolling(window=7).std()

    df_ml['Rolling_Delta'] = df_ml['Delta'].rolling(window=3).mean()

    df_ml['DayOfWeek'] = df_ml.index.dayofweek

    df_ml.dropna(inplace=True)

    features = ['t-1','t-2','SMA_7','Std_7','Rolling_Delta','DayOfWeek']

    X = df_ml[features]
    y = df_ml['Delta']

    split = int(len(df_ml) * 0.8)

    X_train = X.iloc[:split]
    X_test = X.iloc[split:]

    y_train = y.iloc[:split]
    y_test = y.iloc[split:]

    model = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        objective='reg:squarederror'
    )

    model.fit(X_train, y_train)

    preds_delta = model.predict(X_test)

    preds_price = X_test['t-1'].values + preds_delta

    rmse = np.sqrt(mean_squared_error(df_ml['Close'].iloc[split:], preds_price))

    print("RMSE XGBoost :", rmse)

    return model