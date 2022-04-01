import numpy
import pandas as pd
import seaborn
import matplotlib.pyplot as plt
import matplotlib
import math
import os
from warmup.data import x, y1, y2, y3, y4, x4
from common.functions import file_dir, root_dir, absolute_path

if __name__ != "__main__":
    exit(1)


def base_statistics(df: pd.DataFrame) -> pd.DataFrame:
    stats = df.describe().loc[['std', 'mean']]
    stats.loc['var'] = df.var()
    return stats


def pearson_correlation(df: pd.DataFrame) -> pd.DataFrame:
    return df.corr(method='pearson')


def calculate_col_position(iteration_number: int, ncols: int) -> int:
    return math.floor((iteration_number + ncols) % ncols)


def calculate_row_position(iteration_number: int, ncols: int) -> int:
    return math.floor((iteration_number) / (ncols))


def calculate_xaxis_lim(df_index: pd.DataFrame) -> list:
    range_val = df_index.max() - df_index.min()
    return [df_index.min() - range_val * 0.1, df_index.max() + range_val * 0.1]


def calculate_yaxis_lim(df: pd.DataFrame) -> list:
    range_val = df.max().max() - df.min().min()
    return [df.min().min() - range_val * 0.3, df.max().max() + range_val * 0.3]


def get_ax(axs: numpy.ndarray, iteration_number: int,
           ncols: int, nrows: int) -> plt.Axes:
    if nrows > 1 and ncols > 1:
        return axs[calculate_row_position(iteration_number, ncols),
                   calculate_col_position(iteration_number, ncols)]
    return axs[calculate_col_position(iteration_number, ncols * nrows)]


def count_rows_number(df: pd.DataFrame, ncols: int) -> int:
    return math.ceil(df.shape[1] / ncols)


def calculate_figsize(ncols: int, nrows: int) -> tuple:
    return (5 * ncols, 5 * nrows)



def create_dirs(dirs: list, root = root_dir()):
    for dir in dirs:
        os.makedirs(os.path.join(root, dir), exist_ok=True)


def plot(df: pd.DataFrame, ncols: int, nrows: int) -> matplotlib.figure.Figure:
    fig, axs = plt.subplots(ncols=ncols, nrows=nrows)
    df_index = dfAll.reset_index()
    for i in range(0, df.columns.__len__()):
        ax = get_ax(axs, i, N_COLS, n_rows)
        seaborn.regplot(x='index', y=dfAll.columns[i], data=df_index, ax=ax)
        ax.set(xlabel='X', xlim=calculate_xaxis_lim(df_index['index']),
               ylim=calculate_yaxis_lim(df))
    return fig


CSV_DIR = 'csv'
PLOT_DIR = 'plot'
N_COLS = 2
FILE_DIR = file_dir(__file__)
create_dirs([CSV_DIR, PLOT_DIR], root=file_dir(__file__))

df = pd.DataFrame({'y1': y1, 'y2': y2, 'y3': y3}, x)
df4 = pd.DataFrame({'y4': y4}, x4)
dfAll = pd.concat([df, df4])

n_rows = count_rows_number(dfAll, N_COLS)
seaborn.set(rc={"figure.figsize": calculate_figsize(N_COLS, n_rows)})

base_statistics(dfAll).round(2)\
    .to_csv(absolute_path(CSV_DIR, 'statistics.csv', root=FILE_DIR))

pearson_correlation(df).round(2)\
    .to_csv(absolute_path(CSV_DIR, 'pearson.csv', root=FILE_DIR))

plot(df4, N_COLS, n_rows)\
    .savefig(absolute_path(PLOT_DIR, 'plots.jpg', root=FILE_DIR))

# Simple alternative
# plot = seaborn.pairplot(dfAllColIndex,x_vars='index',y_vars=dfAll.columns,
#                         kind='reg',plot_kws={'line_kws':{'color':'red'}})
