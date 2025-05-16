import pandas as pd
import numpy as np
from typing import List


def add_lags(data: pd.DataFrame, lags: List[int] = [1, 7, 14]) -> pd.DataFrame:
    # Add lagged values
    col = data.columns[0]
    for lag in lags:
        data[f"{col}_lag_{lag}"] = data[col].shift(lag)
    return data


def add_rolling_stats(data: pd.DataFrame, windows: List[int] = [7, 14, 30]) -> pd.DataFrame:
    # Add rolling mean and std
    col = data.columns[0]
    for win in windows:
        data[f"{col}_roll_mean_{win}"] = data[col].rolling(win).mean()
        data[f"{col}_roll_std_{win}"] = data[col].rolling(win).std()
    return data


def add_returns(data: pd.DataFrame, use_log: bool = False) -> pd.DataFrame:
    # Add % or log returns
    col = data.columns[0]
    if use_log:
        data[f"{col}_log_return"] = np.log(data[col] / data[col].shift(1))
    else:
        data[f"{col}_pct_return"] = data[col].pct_change()
    return data


def add_time_info(data: pd.DataFrame) -> pd.DataFrame:
    # Add calendar features
    data["dayofweek"] = data.index.dayofweek
    data["month"] = data.index.month
    data["quarter"] = data.index.quarter
    data["year"] = data.index.year
    return data


def z_score(data: pd.DataFrame, exclude: List[str] = []) -> pd.DataFrame:
    # Normalize numeric columns
    numeric = data.select_dtypes(include=[np.number])
    for col in numeric.columns.difference(exclude):
        mean = data[col].mean()
        std = data[col].std()
        data[f"{col}_z"] = (data[col] - mean) / std
    return data


def build_features(data: pd.DataFrame,
                lags: List[int] = [1, 7, 14],
                windows: List[int] = [7, 14, 30],
                log_returns: bool = False,
                normalize: bool = False) -> pd.DataFrame:
    # Combine all feature types
    data = add_lags(data, lags)
    data = add_rolling_stats(data, windows)
    data = add_returns(data, use_log=log_returns)
    data = add_time_info(data)
    if normalize:
        data = z_score(data, exclude=[data.columns[0]])
    return data.dropna()


if __name__ == "__main__":
    import os

    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, "goldprices.csv")

    try:
        raw = pd.read_csv(path, index_col=0, parse_dates=True).sort_index().dropna()
        print("Original:")
        print(raw.head(), "\n")

        features = build_features(raw, log_returns=True, normalize=True)
        print("With Features:")
        print(features.head())

    except Exception as err:
        print(f"Feature error: {err}")
