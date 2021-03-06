import pandas as pd
import numpy as np
import sys
from common.functions import mae, fit_transform, get_imputer,\
    load_data, nan_strategy_methods


def count_nan(df: pd.DataFrame) -> pd.DataFrame:
    return df.isna().sum().loc[lambda x: x > 0]


def percent_nan(df : pd.DataFrame) -> pd.Series:
    nan_cnt = count_nan(df)
    percent_df = pd.Series(dtype='float64')
    for column in nan_cnt.index:
        percent_df[column] = 100 * nan_cnt[column]/df.count().max()
    return percent_df.round(2)


def get_col_config() -> dict:
    return {
        'dtype': {
            'BuildingArea': np.float64,
            'Rooms': np.int8,
            'Price': np.float64,
            'YearBuilt': np.float64
        }
    }


if len(sys.argv) > 1 and sys.argv[1] in nan_strategy_methods():
    nan_strategy = sys.argv[1]
else:
    nan_strategy = 'KNN'


all_df = load_data(config={})
nan = percent_nan(all_df)
print("Procent nan\n", nan)

df = load_data(config=get_col_config()).select_dtypes(include=np.number)

pearson = df.corr(method='pearson')['Price']
print("\nPearson", pearson)

df_wihout_nan = fit_transform(df, get_imputer(nan_strategy))
print("\nMae", mae(df_wihout_nan, ['Car', 'Bathroom', 'Bedroom2'], 'Price'))
