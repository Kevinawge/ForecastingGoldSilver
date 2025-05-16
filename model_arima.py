import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error


def train(data: pd.DataFrame, order=(5, 1, 0)):
    # Fit ARIMA model
    series = data.iloc[:, 0].dropna()
    model = ARIMA(series, order=order)
    return model.fit()


def forecast(model, steps: int) -> pd.DataFrame:
    # Forecast with confidence intervals
    pred = model.get_forecast(steps=steps)
    mean = pred.predicted_mean
    ci = pred.conf_int()

    last_index = model.data.row_labels[-1]
    if isinstance(last_index, pd.Timestamp):
        future = pd.date_range(start=last_index + pd.Timedelta(days=1), periods=steps, freq='D')
        mean.index = future
        ci.index = future

    return pd.DataFrame({
        "forecast": mean,
        "lower": ci.iloc[:, 0],
        "upper": ci.iloc[:, 1]
    })


def split(data: pd.DataFrame, test_size: int = 30):
    return data.iloc[:-test_size], data.iloc[-test_size:]


def evaluate(actual: pd.Series, predicted: pd.Series) -> dict:
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape}


def plot(train: pd.Series, test: pd.Series, result: pd.DataFrame,
        label: str, save: bool = True, save_dir: str = "GoldSilver/figures"):
    # Plot forecast with CI
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(12, 6))
    plt.plot(train, label="Train", linewidth=2)
    plt.plot(test, label="Test", linewidth=2)
    plt.plot(result.index, result["forecast"], label="Forecast", linestyle="--", color="green", linewidth=2)
    plt.fill_between(result.index, result["lower"], result["upper"], color="green", alpha=0.2, label="95% CI")

    xlim = train.index.union(test.index).union(result.index)
    plt.xlim(xlim.min(), xlim.max())
    plt.title(f"ARIMA Forecast – {label}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    if save:
        name = f"arima_{label.lower()}_forecast.png"
        plt.savefig(os.path.join(save_dir, name))

    plt.show(block=False)
    plt.pause(2)
    plt.close()


# Run ARIMA on both series
if __name__ == "__main__":
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_dir = os.path.join(base, "figures")

    for name, file in [("Gold", "goldprices.csv"), ("Silver", "silverprices.csv")]:
        print(f"\nRunning ARIMA for {name}")
        path = os.path.join(os.path.dirname(__file__), file)
        data = pd.read_csv(path, index_col=0, parse_dates=True).sort_index().dropna()

        train_data, test_data = split(data, test_size=30)
        model = train(train_data, order=(5, 1, 0))
        result = forecast(model, steps=len(test_data))

        scores = evaluate(test_data.iloc[:, 0], result["forecast"])
        print(f"{name} Metrics:", scores)

        plot(train_data.iloc[:, 0], test_data.iloc[:, 0], result, label=name, save=True, save_dir=out_dir)
