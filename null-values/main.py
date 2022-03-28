import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.impute._base import _BaseImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
import sys


def count_nan(df : pd.DataFrame) ->pd.DataFrame:
    return df.isna().sum().loc[lambda x : x > 0]


def percent_nan(df : pd.DataFrame) -> pd.Series:
    nanCnt = count_nan(df)
    percentDf = pd.Series(dtype='float64')
    for column in nanCnt.index:
        percentDf[column] = 100 * nanCnt[column]/df.count().max()
    return percentDf.round(2)


def score_dataset(X_train, X_test, y_train, y_test) ->float:
    reg = RandomForestRegressor()
    reg.fit(X_train, y_train)
    return mean_absolute_error(y_test, reg.predict(X_test))


def get_col_config() -> dict:
    return {
        'dtype' : {
            'BuildingArea': np.float64,
            'Rooms': np.int8,
            'Price': np.float64
        },
        'usecols' : ['Rooms','Car','BuildingArea','CouncilArea','YearBuilt','Price']
    }


def get_inputer(strategy='KNN') ->_BaseImputer:
    if strategy.upper() != 'KNN':
        return SimpleImputer(missing_values=np.nan, strategy=strategy)
    else :
        return KNNImputer()


def fit_transform(df,imputer) ->pd.DataFrame:
    nan_cols = df.columns[df.isna().any()].tolist()
    df_imputer = df
    df_imputer[nan_cols] = imputer.fit_transform(df[nan_cols])
    return df_imputer


def load_data(filename = 'houses_data.csv', config=get_col_config()):
    return pd.read_csv(filename, **config)



if len(sys.argv) > 1 and sys.argv[1] in ['mean','median','most_frequent']:
    strategy = sys.argv[1]
else:
    strategy = 'KNN'


all_df = load_data(config={})
nan = percent_nan(all_df)

df = load_data(config=get_col_config()).select_dtypes(include=np.number)
pearson = df.corr(method='pearson')['Price']

df_wihout_nan = fit_transform(df, get_inputer(strategy))

X = df_wihout_nan[['Car', 'BuildingArea', 'YearBuilt']]
y = df_wihout_nan['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
result = score_dataset(X_train, X_test, y_train, y_test)

print(result)



