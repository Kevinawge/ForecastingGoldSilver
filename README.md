## Gold and Silver Time Series Forecasting

# A Comparative Study of Classical and Modern Forecasting Models on Precious Metal Prices

This project investigates the daily price trends of gold and silver from 2013 to 2023 using time series forecasting techniques. Drawing from a decade of historical data, the analysis aims to model and predict future commodity prices using two well-established approaches: the classical ARIMA model and Facebook’s modern Prophet framework. These models were selected for their widespread use in financial forecasting, with ARIMA offering robustness for stationary patterns and Prophet providing flexibility in capturing trend and seasonality.

The workflow begins with a structured data preprocessing pipeline, which standardizes dates, fills missing values, and engineers temporal features such as lagged variables and normalized components. Both gold and silver datasets are subjected to exploratory data analysis to assess volatility, trend behavior, and mean-reversion tendencies. Following this, each model is trained and tested on a rolling 30-day holdout window to evaluate out-of-sample performance using RMSE, MAE, and MAPE as scoring metrics.

Visual diagnostics including forecast overlays, confidence intervals, and residual error distributions supplement the evaluation. Results indicate that ARIMA consistently delivered lower error rates across both assets, outperforming Prophet in short-term prediction accuracy. The project demonstrates the enduring value of well-calibrated statistical models in financial time series analysis and emphasizes the importance of model interpretability alongside predictive accuracy.

Source[https://www.kaggle.com/datasets/kapturovalexander/gold-and-silver-prices-2013-2023]
