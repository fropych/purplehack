from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Dataset


def get_data(
    train_path: str = "data/train_ae.pqt",
    test_path: str = "data/test_ae.pqt",
    catigorical_cols: Optional[List[str]] = None,
):
    """
    Load and preprocess train and test data.

    Args:
        train_path (str): Path to the training data.
        test_path (str): Path to the test data.
        catigorical_cols (Optional[List[str]]): List of categorical columns.
            If None, default columns are used.

    Returns:
        X (np.ndarray): Preprocessed data.
        num_categorical_features (int): Number of categorical features.
        num_numerical_features (int): Number of numerical features.
    """
    if catigorical_cols is None:
        catigorical_cols = [
            "channel_code",
            "okved",
            "segment",
            "city",
            "start_cluster",
            "index_city_code",
            "ogrn_month",
            "ogrn_year",
        ]

    train_data = pd.read_parquet(train_path).drop(
        [
            "id",
            "date",
            "end_cluster",
        ],
        axis=1,
    )
    test_data = pd.read_parquet(test_path)
    test_data = test_data.drop(
        [
            "id",
            "date",
        ],
        axis=1,
    )

    X_nums = np.vstack(
        [
            train_data.select_dtypes(exclude=["object"]).fillna(
                train_data.select_dtypes(exclude=["object"]).mean()
            ),
            test_data.select_dtypes(exclude=["object"]).fillna(
                test_data.select_dtypes(exclude=["object"]).mean()
            ),
        ]
    )
    X_nums = (X_nums - X_nums.mean(0)) / X_nums.std(0)

    X_cat = np.vstack([train_data[catigorical_cols], test_data[catigorical_cols]])
    encoder = OneHotEncoder(sparse_output=False)
    X_cat = encoder.fit_transform(X_cat)
    X = np.concatenate([X_cat, X_nums], axis=1)
    return X, X_cat.shape[1], X_nums.shape[1]


class SingleDataset(Dataset):
    def __init__(self, x: np.ndarray, is_sparse: bool = False) -> None:
        """
        Initialize a SingleDataset object.

        Args:
            x (np.ndarray): The data to be stored in the dataset.
            is_sparse (bool, optional): Whether the data is in sparse format. Defaults to False.
        """
        self.x = x.astype("float32")
        self.is_sparse = is_sparse

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        x = self.x[index]
        if self.is_sparse:
            x = x.toarray().squeeze()
        return x
