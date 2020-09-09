import os
import pickle


def preprocess_df(df, data_dir):
    df["path"] = df.UID.map(lambda x: os.path.join(data_dir, f"{x}.jpeg"))
    return df


def load_from_file_fast(file_name):
    with open(file_name, "rb") as f:
        return pickle.load(f)


def save_in_file_fast(arr, file_name):
    with open(file_name, "wb") as f:
        pickle.dump(arr, f, protocol=4)
