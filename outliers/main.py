import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import median_abs_deviation
from common.functions import mae, fit_transform, get_imputer, load_data, nan_strategy_methods


def iqr(column : pd.Series) -> pd.Series:
    q1 = column.quantile(.25)
    q3 = column.quantile(.75)
    return (column >= (q1 - q1 * 1.5)) & (column <= (q3 + q3 * 1.5))


def percentile_01_09(column : pd.Series) -> pd.Series:
    return (column >= column.quantile(.1)) & (column <= column.quantile(.9))


def log_transform(df: pd.DataFrame) -> pd.DataFrame:
    df_new= pd.DataFrame()
    for col in df.columns:
        df_new[col] = df[col].map(lambda i: np.log(i) if i > 0 else 0)
    return df_new


def z_score(column : pd.Series) -> pd.Series:
    return np.abs(column - column.mean()) <= (3 * column.std())


def modified_z_score_outlier(column: pd.Series) -> pd.Series:
    mad_column = median_abs_deviation(column)
    median = np.median(column)
    mad_score = np.abs(0.6745 * (column - median) / mad_column)
    return mad_score <= 3.5


def get_args() -> tuple:
    if len(sys.argv) > 1 and sys.argv[1] in nan_strategy_methods():
        nan_strategy = sys.argv[1]
    else:
        nan_strategy = None

    separate_column = True if ((len(sys.argv) > 2 and sys.argv[2] == '1') or
                               (len(sys.argv) < 3 and nan_strategy is None)) else False

    log_transfom_enable = True if len(sys.argv) > 3 and sys.argv[3] == '1' else False

    return (nan_strategy, separate_column,log_transfom_enable)


nan_strategy, separate_column, log_transfom_enable = get_args()

df = load_data(config={"usecols": ['YearBuilt', 'BuildingArea', 'Price', 'Car']})
df = fit_transform(df, get_imputer(nan_strategy)) if nan_strategy is not None else df
df = log_transform(df) if log_transfom_enable else df

print(df.describe())
df.boxplot()
plt.show()


Y_COL = 'Price'
rows_cnt = df.shape[0]
for func in [iqr, percentile_01_09, modified_z_score_outlier, z_score]:
    print("\nFunction " + func.__name__)
    mask = df.apply(lambda x: func(x))

    print("\nNumber of outliers:")
    print(mask.apply(lambda x: rows_cnt - x.sum()), "\n")

    x_cols = df.drop([Y_COL], axis=1).columns
    if separate_column:
        for col in x_cols:
            print(col + " Mae: " + str(mae(df[mask], [col], Y_COL)))

    print("\nMae: " + str(mae(df[mask], x_cols, Y_COL)))



