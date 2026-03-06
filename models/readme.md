🚀 Bitcoin Price Prediction – Hybrid ML & Deep Learning



A hybrid artificial intelligence system for Bitcoin price forecasting, combining Statistical Modeling, Machine Learning, and Deep Learning.

The project compares ARIMA, XGBoost, and LSTM models and deploys predictions through a FastAPI real-time API.

📌 Project Overview

Bitcoin prices exhibit:

High volatility

Non-stationary time series

Sudden market shocks

This project evaluates three complementary approaches:

Approach	Model	Goal
Statistical	ARIMA	Baseline time-series modeling
Machine Learning	XGBoost	Predict short-term variations
Deep Learning	LSTM	Capture long-term temporal patterns

The final goal is to produce a robust prediction signal using a hybrid modeling strategy.

📸 Prediction Visualization

Insert here the real vs predicted price graph.

Example:

![Bitcoin Prediction](images/prediction_plot.png)

Recommended graph:

Blue line → Real price

Red line → Model prediction

🧠 Methodology
1️⃣ ARIMA — Statistical Model

Traditional time series forecasting method.

Advantages

Interpretable

Good for stationary series

Limitations

Struggles with explosive crypto trends

Sensitive to non-stationarity

Conclusion: limited performance on cryptocurrency markets.

2️⃣ XGBoost — Machine Learning Model

We model the price variation (Δ price) instead of the raw price.

Advantages:

Strong performance on tabular data

Captures micro-market movements

📊 Result:

65% directional accuracy

3️⃣ LSTM — Deep Learning Model

Long Short-Term Memory networks learn temporal dependencies in financial time series.

Configuration:

Sequence window: 30 days

Sequential training

Advantages:

Captures long-term market cycles

Handles non-linear dynamics

🤝 Model Consensus

Predictions from multiple models are combined to produce a more robust trading signal.

This approach reduces the risk of relying on a single model.

💻 Tech Stack
Programming Language

Python

Data Science Libraries

Pandas

NumPy

Scikit-Learn

XGBoost

TensorFlow / Keras

Visualization

Matplotlib

Plotly

Deployment

FastAPI

Uvicorn

🏗️ Project Architecture
BTC-Prediction
│
├── data
│   └── bitcoin_price.csv
│
├── notebooks
│   └── btc_analysis.ipynb
│
├── models
│   ├── arima_model.pkl
│   ├── xgboost_model.pkl
│   └── lstm_model.h5
│
├── api
│   └── main.py
│
├── images
│   └── prediction_plot.png
│
├── requirements.txt
│
└── README.md
🚀 API Deployment

The project includes a FastAPI server providing real-time predictions.

Start the API
uvicorn main:app --reload

API documentation available at:

http://127.0.0.1:8000/docs
📡 Example API Response
{
  "model": "XGBoost",
  "prediction": 64521.32,
  "confidence": 0.65
}
📈 Key Results
Model	RMSE	Strength
XGBoost	~1771 USD	Sensitive to short-term fluctuations
LSTM	~3823 USD	Captures long-term patterns
📂 Installation
Clone the repository
git clone https://github.com/yourusername/BTC-Prediction.git
Install dependencies
pip install -r requirements.txt
▶️ Run the Analysis

Launch the notebook:

btc_analysis.ipynb
🔮 Future Improvements

Transformer models for time series

Integration of macro-economic indicators

Backtesting trading strategies

Docker deployment

Cloud deployment (AWS / GCP)

👨‍💻 Author

Kevin Gordan Njike Njingang

Machine Learning Engineer
Artificial Intelligence Researcher

⭐ Support

If you like this project:

⭐ Star the repository
🍴 Fork it
📢 Share it