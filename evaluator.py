import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error


def evaluate(actual: pd.Series, predicted: pd.Series) -> dict:
    # Compute RMSE, MAE, MAPE
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape}


def summarize(metrics: dict) -> pd.DataFrame:
    # Summarize metric results
    return pd.DataFrame(metrics).T.round(3)


def compare_plot(actual: pd.Series, forecasts: dict,
                title="Forecast Comparison", save_dir="GoldSilver/figures", filename=None):
    # Plot actual vs. model forecasts
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(12, 6))
    plt.plot(actual, label="Actual", linewidth=2)

    for name, series in forecasts.items():
        plt.plot(series.index, series.values, label=name, linestyle="--")

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    if filename:
        plt.savefig(os.path.join(save_dir, filename))

    plt.show(block=False)
    plt.pause(2)
    plt.close()


def error_plot(actual: pd.Series, predicted: pd.Series,
            label="Model", save_dir="GoldSilver/figures", filename=None):
    # Plot forecast error distribution
    os.makedirs(save_dir, exist_ok=True)
    err = actual - predicted

    plt.figure(figsize=(10, 5))
    plt.hist(err, bins=30, color="slateblue", alpha=0.7)
    plt.axvline(0, color="black", linestyle="--")
    plt.title(f"Error Distribution – {label}")
    plt.xlabel("Error")
    plt.ylabel("Count")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    if filename:
        plt.savefig(os.path.join(save_dir, filename))

    plt.show(block=False)
    plt.pause(2)
    plt.close()

