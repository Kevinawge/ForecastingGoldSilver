import pandas as pd
from typing import Tuple, Optional


def load_series(path: str, date_col: str = None,
                value_col: Optional[str] = None,
                freq: Optional[str] = None) -> pd.DataFrame:
    # Load and prepare a single time series
    data = pd.read_csv(path)

    if date_col is None:
        date_col = data.columns[0]

    data[date_col] = pd.to_datetime(data[date_col], errors='coerce')
    data = data.set_index(date_col)

    if value_col:
        data = data[[value_col]]
    else:
        data = data.iloc[:, [0]]

    data = data.sort_index()
    data = data[~data.index.duplicated(keep="first")]
    data = data.dropna()

    if freq:
        data = data.asfreq(freq).interpolate(method="linear")

    return data


def load_data(gold_path: str, silver_path: str,
            date_col: str = None,
            gold_col: Optional[str] = None,
            silver_col: Optional[str] = None,
            freq: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Load and return both gold and silver series
    gold = load_series(gold_path, date_col, gold_col, freq)
    silver = load_series(silver_path, date_col, silver_col, freq)
    return gold, silver


if __name__ == "__main__":
    import os

    base_dir = os.path.dirname(os.path.abspath(__file__))
    gold_path = os.path.join(base_dir, "goldprices.csv")
    silver_path = os.path.join(base_dir, "silverprices.csv")

    try:
        gold, silver = load_data(gold_path, silver_path)
        print("Gold sample:")
        print(gold.head(), "\n")
        print("Silver sample:")
        print(silver.head())
    except Exception as err:
        print(f"Data loading error: {err}")

