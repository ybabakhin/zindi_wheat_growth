import pickle
from typing import Any, List, Optional

import glob
import os
import pandas as pd
import pytorch_lightning as pl


def preprocess_df(df: pd.DataFrame, data_dir: str) -> pd.DataFrame:
    """Adds path to image names"""

    df["path"] = df.UID.map(lambda x: os.path.join(data_dir, f"{x}.jpeg"))
    return df


def combine_dataframes(
    models_list: List[int],
    logs_dir: str,
    filename: str,
    output_colname: str = "growth_stage",
    agg_func: Optional[str] = "mean",
) -> pd.DataFrame:
    """Combines multiple model predictions for the ensemble.

    Args:
        models_list: list of model IDs to be ensembled
        logs_dir: directory with model logs
        filename: file name of the predictions
        output_colname: column name for the ensembled predictions
        agg_func: one of {'mean', 'mode', None}. Aggregation method for the ensembling.
            If None, then no aggregation is applied

    Returns:
        DataFrame with image ID and ensembled prediction
    """

    predictions = [
        pd.read_csv(os.path.join(logs_dir, f"model_{m_id}", filename), index_col="UID")
        for m_id in models_list
    ]

    predictions = pd.concat(predictions, axis=1)
    if agg_func is None:
        return predictions

    if agg_func == "mean":
        predictions = predictions.mean(axis=1).reset_index()
    elif agg_func == "mode":
        predictions = predictions.mode(axis=1)[0].reset_index()

    predictions.columns = ["UID", output_colname]

    return predictions


def get_single_model_path(checkpoints_dir: str):
    """Lightning's hack to load last epoch model if available.

    Args:
        checkpoints_dir: directory with the image checkpoints

    Returns:
        Path to a single checkpoint to be loaded
    """

    last_path = os.path.join(checkpoints_dir, "last.ckpt")
    if os.path.exists(last_path):
        return last_path

    return glob.glob(os.path.join(checkpoints_dir, "*.ckpt"))[0]


def setup_environment(seed: int, gpu_list: List) -> None:
    """Sets up environment variables

    Args:
        seed: random seed
        gpu_list: list of GPUs available for the experiment
    """

    os.environ["HYDRA_FULL_ERROR"] = "1"
    pl.seed_everything(seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(x) for x in gpu_list])


def load_from_file_fast(file_name: str) -> Any:
    """Loads pickled file"""

    with open(file_name, "rb") as f:
        return pickle.load(f)


def save_in_file_fast(arr: Any, file_name: str) -> None:
    """Pickles objects to files"""

    with open(file_name, "wb") as f:
        pickle.dump(arr, f, protocol=4)
