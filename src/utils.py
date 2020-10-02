import pickle
from typing import Any, List

import os
import pandas as pd


def preprocess_df(df: pd.DataFrame, data_dir: str) -> pd.DataFrame:
    df["path"] = df.UID.map(lambda x: os.path.join(data_dir, f"{x}.jpeg"))
    return df


def combine_dataframes(
    models_list: List[int], logs_dir: str, filename: str, output_colname: str
) -> pd.DataFrame:
    predictions = [
        pd.read_csv(os.path.join(logs_dir, f"model_{m_id}", filename), index_col="UID")
        for m_id in models_list
    ]

    predictions = pd.concat(predictions, axis=1)
    predictions = predictions.mean(axis=1).reset_index()
    predictions.columns = ["UID", output_colname]

    return predictions


def load_from_file_fast(file_name: str) -> Any:
    with open(file_name, "rb") as f:
        return pickle.load(f)


def save_in_file_fast(arr: Any, file_name: str) -> None:
    with open(file_name, "wb") as f:
        pickle.dump(arr, f, protocol=4)
