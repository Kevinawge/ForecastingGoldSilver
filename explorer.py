import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose


def _make_dir(path: str):
    # Create directory if it doesn't exist
    if path and not os.path.exists(path):
        os.makedirs(path)


def plot_dual_prices(gold, silver, save=False, save_dir="GoldSilver/figures"):
    # Plot gold and silver together on dual y-axes
    _make_dir(save_dir)
    fig, ax1 = plt.subplots(figsize=(14, 6))

    ax1.set_title("Gold and Silver Prices")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Gold", color='gold')
    ax1.plot(gold.index, gold.iloc[:, 0], color='gold', linewidth=2)
    ax1.tick_params(axis='y', labelcolor='gold')

    ax2 = ax1.twinx()
    ax2.set_ylabel("Silver", color='silver')
    ax2.plot(silver.index, silver.iloc[:, 0], color='grey', linewidth=2)
    ax2.tick_params(axis='y', labelcolor='grey')

    fig.tight_layout()

    if save:
        plt.savefig(os.path.join(save_dir, "dual_prices.png"))
    plt.close()


def plot_rolling_statistics(data, window=30, label="Series", save=False, save_dir="GoldSilver/figures"):
    # Plot rolling mean and std
    _make_dir(save_dir)
    series = data.iloc[:, 0]
    roll_mean = series.rolling(window).mean()
    roll_std = series.rolling(window).std()

    plt.figure(figsize=(14, 6))
    plt.plot(series, alpha=0.5, label=f"{label}")
    plt.plot(roll_mean, color='red', label=f"Rolling Mean ({window}d)")
    plt.fill_between(series.index, roll_mean - 2 * roll_std, roll_mean + 2 * roll_std,
                    color='orange', alpha=0.3, label='±2 Std Dev')
    plt.title(f"{label} Rolling Stats")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    if save:
        name = f"{label.lower()}_rolling_stats.png"
        plt.savefig(os.path.join(save_dir, name))
    plt.close()


def plot_seasonal_decompose(data, model='additive', freq=30, label="Series", save=False, save_dir="GoldSilver/figures"):
    # Plot seasonal decomposition
    _make_dir(save_dir)
    series = data.iloc[:, 0].dropna()
    result = seasonal_decompose(series, model=model, period=freq)

    fig = result.plot()
    fig.set_size_inches(14, 10)
    fig.suptitle(f"{label} Decomposition", fontsize=16)
    plt.tight_layout()

    if save:
        name = f"{label.lower()}_decomposition.png"
        fig.savefig(os.path.join(save_dir, name))
    plt.close()


def plot_return_distribution(data, label="Series", save=False, save_dir="GoldSilver/figures"):
    # Plot distribution of daily returns
    _make_dir(save_dir)
    returns = data.pct_change().dropna()

    plt.figure(figsize=(10, 5))
    sns.histplot(returns.iloc[:, 0], bins=50, kde=True, color='teal')
    plt.title(f"{label} Return Distribution")
    plt.xlabel("Daily Return")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    if save:
        name = f"{label.lower()}_returns_hist.png"
        plt.savefig(os.path.join(save_dir, name))
    plt.close()


def plot_acf_pacf(data, lags=40, label="Series", save=False, save_dir="GoldSilver/figures"):
    # Plot autocorrelation and partial autocorrelation
    _make_dir(save_dir)
    series = data.iloc[:, 0].dropna()

    fig, ax = plt.subplots(2, 1, figsize=(12, 8))
    plot_acf(series, lags=lags, ax=ax[0])
    ax[0].set_title(f"{label} ACF")
    plot_pacf(series, lags=lags, ax=ax[1])
    ax[1].set_title(f"{label} PACF")
    plt.tight_layout()

    if save:
        name = f"{label.lower()}_acf_pacf.png"
        fig.savefig(os.path.join(save_dir, name))
    plt.close()


#Test Run
if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    gold_path = os.path.join(base, "goldprices.csv")
    silver_path = os.path.join(base, "silverprices.csv")
    out_dir = os.path.join(base, "figures")

    try:
        gold = pd.read_csv(gold_path, index_col=0, parse_dates=True).sort_index().dropna()
        silver = pd.read_csv(silver_path, index_col=0, parse_dates=True).sort_index().dropna()

        plot_dual_prices(gold, silver, save=True, save_dir=out_dir)
        plot_rolling_statistics(gold, label="Gold", save=True, save_dir=out_dir)
        plot_seasonal_decompose(gold, label="Gold", freq=30, save=True, save_dir=out_dir)
        plot_return_distribution(gold, label="Gold", save=True, save_dir=out_dir)
        plot_acf_pacf(gold, label="Gold", save=True, save_dir=out_dir)

    except Exception as err:
        print(f"Plotting failed: {err}")

