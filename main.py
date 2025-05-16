import os
import pandas as pd
from data_loader import load_data
from model_arima import train as train_arima, forecast as forecast_arima, split as split_arima
from model_prophet import train as train_prophet, forecast as forecast_prophet, split as split_prophet
from evaluator import evaluate, summarize, compare_plot, error_plot


def run_pipeline(name: str, data: pd.DataFrame, test_size=30, save_dir="GoldSilver/figures"):
    # Run ARIMA and Prophet forecasting + evaluation
    print(f"\nRunning forecast for {name}")

    # ARIMA
    train_a, test_a = split_arima(data, test_size)
    model_a = train_arima(train_a)
    pred_a = forecast_arima(model_a, steps=test_size)["forecast"]

    # Prophet
    train_p, test_p = split_prophet(data, test_size)
    model_p = train_prophet(train_p)
    pred_p = forecast_prophet(model_p, periods=test_size)["yhat"]

    actual = test_a.iloc[:, 0]
    scores = {
        "ARIMA": evaluate(actual, pred_a),
        "Prophet": evaluate(actual, pred_p)
    }

    print(summarize(scores))

    compare_plot(actual, {
        "ARIMA": pred_a,
        "Prophet": pred_p
    }, title=f"{name} Forecast Comparison",
    save_dir=save_dir,
    filename=f"{name.lower()}_comparison.png")

    error_plot(actual, pred_a, label=f"{name} ARIMA",
            save_dir=save_dir, filename=f"{name.lower()}_arima_errors.png")

    error_plot(actual, pred_p, label=f"{name} Prophet",
            save_dir=save_dir, filename=f"{name.lower()}_prophet_errors.png")


if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    gold_path = os.path.join(base, "goldprices.csv")
    silver_path = os.path.join(base, "silverprices.csv")

    gold, silver = load_data(gold_path, silver_path)

    run_pipeline("Gold", gold)
    run_pipeline("Silver", silver)

    print("\nDone.")

