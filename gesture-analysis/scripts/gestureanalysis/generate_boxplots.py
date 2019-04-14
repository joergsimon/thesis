from __future__ import annotations
from typing import List, Callable
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from . import utils
from . import specific_utils as sutils


def print_valurange_summary(onebigline: np.array, username: str):
    n = len(onebigline)
    nh = n // 2
    print('user: ', username, ' vals: ', n, ' mean: ', onebigline.mean(), ' std: ', onebigline.std(), ' min: ',
          onebigline.min(), ' max: ', onebigline.max())
    print(onebigline[:4], "...", onebigline[nh:nh + 4], "...", onebigline[-4:])
    print("")


def show_valuerange_individual_boxplots(usernames: List[str], users: List, title: str,
                                        column: str, remove_outliers: bool,
                                        higher_percentile: float, lower_percentile: float,
                                        print_summary: bool, show_overal: bool):
    all_vals = []
    lines = sutils.values_per_user(usernames, users, column, remove_outliers,
                                   higher_percentile, lower_percentile, True)
    for onebigline, username in lines:
        if print_summary:
            print_valurange_summary(onebigline, username)
        fig, axes = plt.subplots()
        sns.violinplot(data=pd.DataFrame(data=onebigline.T), ax=axes)
        #pd.DataFrame(data=onebigline.T).boxplot()
        plt.title(f'values of {title} from {username}')
        plt.show()
        all_vals += list(onebigline)
    if show_overal:
        all_vals = np.array(all_vals)
        if print_summary:
            print_valurange_summary(onebigline, 'all_users')
        fig, axes = plt.subplots()
        sns.violinplot(data=pd.DataFrame(data=all_vals).T, ax=axes)
        #pd.DataFrame(data=all_vals).T.boxplot()
        plt.title(f'values of {title} from all users')
        plt.show()


def show_valuerange_boxplots_in_one_image(usernames: List[str], users: List, title: str, column: str,
                                          remove_outliers: bool, higher_percentile: float,
                                          lower_percentile: float, print_summary: bool,
                                          show_overal: bool):
    return show_valuerange_boxplots_in_one_imagefor_label_type_and_gesture(
        usernames, users, title, column, None, 'all_values', remove_outliers,
        higher_percentile, lower_percentile, print_summary, show_overal)


def show_valuerange_boxplots_in_one_imagefor_label_type_and_gesture(
        usernames: List[str], users: List, title: str, column: str,
        label_type: str, gesture: str, remove_outliers: bool,
        higher_percentile: float, lower_percentile: float,
        print_summary: bool, show_overal: bool):
    all_vals = []
    all_dfs = []
    lines = sutils.values_per_user_for_label_type_and_gesture(
        usernames, users, column, label_type, gesture, remove_outliers,
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
    fig, axes = plt.subplots()
    v = sns.violinplot(x="participant ID", y="value", data=df, ax=axes)
    v.set_xticklabels(rotation=45)
    #df.boxplot()
    plt.title(f'values of {title} for all different users')
    plt.show()
    if show_overal:
        all_vals = np.array(all_vals)
        if print_summary:
            print_valurange_summary(onebigline, 'all_users')
        #pd.DataFrame(data=all_vals).T.boxplot()
        v = sns.violinplot(x="participant ID", y="value", data=pd.DataFrame(data=all_vals).T, ax=axes)
        v.set_xticklabels(rotation=45)
        plt.title(f'values of {title} for all different users including overall distribution')
        plt.show()
