import os
from pathlib import Path

import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.impute._base import _BaseImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import numpy as np

def score_dataset(X_train, X_test, y_train, y_test) -> float:
    reg = RandomForestRegressor()
    reg.fit(X_train, y_train)
    return mean_absolute_error(y_test, reg.predict(X_test))


def mae(df : pd.DataFrame, x_cols : list, y_col : str) -> float:
    df_wihout_nan = df[[*x_cols, y_col]].dropna()
    if df_wihout_nan.shape[0] < 1:
        return 0
    X = df_wihout_nan[x_cols]
    y = df_wihout_nan[y_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    return score_dataset(X_train, X_test, y_train, y_test)


def fit_transform(df : pd.DataFrame, imputer : _BaseImputer) -> pd.DataFrame:
    nan_cols = df.columns[df.isna().any()].tolist()
    df_imputer = df
    df_imputer[nan_cols] = imputer.fit_transform(df[nan_cols])
    return df_imputer


def get_imputer(strategy : str = 'KNN') -> _BaseImputer:
    if strategy.upper() != 'KNN':
        return SimpleImputer(missing_values=np.nan, strategy=strategy)
    else :
        return KNNImputer()


def root_dir() -> str:
    return Path(__file__).parent.parent


def file_dir(file : str) -> str:
    return os.path.dirname(os.path.abspath(file))


def absolute_path(*paths : list, root : dict = root_dir()) -> str:
    return os.path.join(root, *paths)


def load_data(filename : str = absolute_path('common', 'houses_data.csv') , config: dict = {}) -> str:
    return pd.read_csv(filename, **config)


def nan_strategy_methods() -> list:
    return ['mean','median','most_frequent', 'KNN']