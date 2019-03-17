from __future__ import annotations
from typing import List, Callable
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from . import utils
from . import specific_utils as sutils


def print_valurange_summary(onebigline: np.array, username: str):
    n = len(onebigline)
    nh = n // 2
    print('user: ', username, ' vals: ', n, ' mean: ', onebigline.mean(), ' std: ', onebigline.std(), ' min: ',
          onebigline.min(), ' max: ', onebigline.max())
    print(onebigline[:4], "...", onebigline[nh:nh + 4], "...", onebigline[-4:])
    print("")


def show_valuerange_individual_boxplots(usernames: List[str], users: List, column: str,
                                        remove_outliers: bool, higher_percentile: float,
                                        lower_percentile: float, print_summary: bool,
                                        show_overal: bool):
    all_vals = []
    lines = sutils.values_per_user(usernames, users, column, remove_outliers,
                                   higher_percentile, lower_percentile, True)
    for onebigline, username in lines:
        if print_summary:
            print_valurange_summary(onebigline, username)
        pd.DataFrame(data=onebigline.T).boxplot()
        plt.show()
        all_vals += list(onebigline)
    if show_overal:
        all_vals = np.array(all_vals)
        if print_summary:
            print_valurange_summary(onebigline, 'all_users')
        pd.DataFrame(data=all_vals).T.boxplot()
        plt.show()


def show_valuerange_boxplots_in_one_image(usernames: List[str], users: List, column: str,
                                          remove_outliers: bool, higher_percentile: float,
                                          lower_percentile: float, print_summary: bool,
                                          show_overal: bool):
    all_vals = []
    all_dfs = []
    lines = sutils.values_per_user(usernames, users, column, remove_outliers,
                                   higher_percentile, lower_percentile, True)
    for onebigline, username in lines:
        if print_summary:
            print_valurange_summary(onebigline, username)
        if show_overal:
            all_vals += list(onebigline)
        df = pd.DataFrame(data=onebigline.T, columns=[username])
        all_dfs.append(df)
    df = pd.concat(all_dfs, axis=1)
    #print(df)
    df.boxplot()
    plt.show()
    if show_overal:
        all_vals = np.array(all_vals)
        if print_summary:
            print_valurange_summary(onebigline, 'all_users')
        pd.DataFrame(data=all_vals).T.boxplot()
        plt.show()
