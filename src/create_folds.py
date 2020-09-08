import pandas as pd
from sklearn.model_selection import StratifiedKFold


def split_data(df):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=24)
    for fold, (idxT, idxV) in enumerate(skf.split(df, df.growth_stage)):
        df.iloc[idxV, df.columns.get_loc("fold")] = fold

    return df


if __name__ == "__main__":
    train = pd.read_csv("../data/Train.csv")
    train["fold"] = -1
    all_train = []

    for label_quality in [1, 2]:
        df = train.loc[train.label_quality == label_quality].copy()
        df = split_data(df)
        all_train.append(df)

    all_train = pd.concat(all_train).sample(frac=1, random_state=13)

    all_train.to_csv("../data/Train_proc.csv", index=False)
