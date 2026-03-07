import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error


def train_arima(df_day):

    train_size = int(len(df_day) * 0.8)

    train_log = df_day['Log_Price'][:train_size]
    test_log = df_day['Log_Price'][train_size:]

    test_actual = df_day['Close'][train_size:]

    model = ARIMA(train_log, order=(5,1,0))

    model_fit = model.fit()

    preds_log = model_fit.forecast(steps=len(test_log))

    preds_final = np.exp(preds_log)

    rmse = np.sqrt(mean_squared_error(test_actual, preds_final))

    print("RMSE ARIMA :", rmse)

    return preds_final