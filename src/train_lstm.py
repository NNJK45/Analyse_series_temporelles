import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


def train_lstm(df_day):

    scaler = MinMaxScaler()

    scaled_data = scaler.fit_transform(df_day[['Close']])

    def create_sequences(data, seq_length=30):

        X, y = [], []

        for i in range(seq_length, len(data)):
            X.append(data[i-seq_length:i,0])
            y.append(data[i,0])

        return np.array(X), np.array(y)


    SEQ_LENGTH = 30

    X, y = create_sequences(scaled_data, SEQ_LENGTH)

    X = X.reshape((X.shape[0], X.shape[1], 1))

    train_size = int(0.8 * len(X))

    X_train = X[:train_size]
    X_test = X[train_size:]

    y_train = y[:train_size]
    y_test = y[train_size:]

    model = Sequential()

    model.add(LSTM(50, return_sequences=True, input_shape=(SEQ_LENGTH,1)))
    model.add(Dropout(0.2))

    model.add(LSTM(50))
    model.add(Dropout(0.2))

    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(X_train, y_train, epochs=20, batch_size=32)

    preds = model.predict(X_test)

    preds = scaler.inverse_transform(preds)

    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1,1))

    rmse = np.sqrt(mean_squared_error(y_test_rescaled, preds))

    print("RMSE LSTM :", rmse)

    return model