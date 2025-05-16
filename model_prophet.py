import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error


def format_for_prophet(data: pd.DataFrame) -> pd.DataFrame:
    # Rename index and value columns for Prophet
    return data.reset_index().rename(columns={data.columns[0]: "y", data.index.name or data.columns[0]: "ds"})


def train(data: pd.DataFrame) -> Prophet:
    # Fit Prophet model
    formatted = format_for_prophet(data)
    model = Prophet()
    model.fit(formatted)
    return model


def forecast(model: Prophet, periods: int, freq: str = "D") -> pd.DataFrame:
    # Predict future values
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    return forecast.set_index("ds")[["yhat", "yhat_lower", "yhat_upper"]].tail(periods)


def split(data: pd.DataFrame, test_size: int = 30):
    return data.iloc[:-test_size], data.iloc[-test_size:]


def evaluate(actual: pd.Series, predicted: pd.Series) -> dict:
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape}


def plot(train, test, result, label="Series", save_dir="GoldSilver/figures", save=True):
    # Plot Prophet forecast
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train.iloc[:, 0], label="Train", linewidth=2)
    plt.plot(test.index, test.iloc[:, 0], label="Test", linewidth=2)
    plt.plot(result.index, result["yhat"], label="Forecast", linestyle="--", color="green", linewidth=2)
    plt.fill_between(result.index, result["yhat_lower"], result["yhat_upper"],
                    color="green", alpha=0.2, label="95% CI")

    xlim = train.index.union(test.index).union(result.index)
    plt.xlim(xlim.min(), xlim.max())
    plt.title(f"Prophet Forecast – {label}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    if save:
        name = f"prophet_{label.lower()}_forecast.png"
        plt.savefig(os.path.join(save_dir, name))

    plt.show(block=False)
    plt.pause(2)
    plt.close()


# ----------------- Run both assets ------------------

if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base, "figures")

    for name, file in [("Gold", "goldprices.csv"), ("Silver", "silverprices.csv")]:
        print(f"\nRunning Prophet for {name}")

        path = os.path.join(base, file)
        data = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()

        train_data, test_data = split(data, test_size=30)
        model = train(train_data)
        result = forecast(model, periods=len(test_data))

        scores = evaluate(test_data.iloc[:, 0], result["yhat"])
        print(f"{name} Metrics:", scores)

        plot(train_data, test_data, result, label=name, save_dir=out_dir)

    print("\nDone.")

