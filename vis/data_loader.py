from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing, fetch_openml
from sklearn.model_selection import train_test_split


class DataSplit(NamedTuple):
    features: pd.DataFrame
    targets: pd.DataFrame


class Dataset(NamedTuple):
    train: DataSplit
    test: DataSplit


def load_cali_housing(random_state: int, test_size: float = .2) -> Dataset:
    x, y = fetch_california_housing(return_X_y=True, as_frame=True)
    y = y * 100000
    return process(x, y, random_state, test_size)


def load_mpg(random_state: int, test_size: float = .2) -> Dataset:
    x, y = fetch_openml(name='autoMpg', version=3, return_X_y=True, as_frame=True)
    return process(x, y, random_state, test_size)


def process(x: np.ndarray, y: np.ndarray, random_state: int, test_size: float) -> Dataset:
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state)
    return Dataset(train=DataSplit(x_train, y_train), test=DataSplit(x_test, y_test))
