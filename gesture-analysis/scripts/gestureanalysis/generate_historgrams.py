from __future__ import annotations
from typing import List
import pathlib
import time
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from . import specific_utils as sutils
from . import image_utils as iutils


def draw_and_save_hist(array, path):
    plt.hist(array)
    plt.savefig(path)
    plt.close()


def hist_with(path: str, label_group: List[sutils.LabelGroup], label_type: str,
              skipp: bool, skippable: iutils.Skippable):
    if skipp:
        if pathlib.Path(path).exists():
            skippable.add_skipped_thing(path, show_message=True)
            return
    deltas = sutils.get_timedeltas(label_group, label_type)
    draw_and_save_hist(list(map(lambda x: x.total_seconds(), deltas)), path)


def get_path(username, gesture, label_type):
    fig_base_path = iutils.img_base_path(username, gesture)
    path = f'{fig_base_path}timing_of_{label_type}.png'
    return path


all_groups = []


def generate_histogram_callback(
        users: List, all_users_key: str, label_type: str,
        collector: iutils.AllUsersCollector, skipp: bool):

    def visualize_channel_gesture_callback(skippable: iutils.Skippable,
                                           username: str, gesture: str,
                                           bar: tqdm.tqdm_notebook):
        global all_groups
        if gesture != collector.current_gesture:
            if collector.current_gesture is not None:
                path = get_path(all_users_key, gesture, label_type)
                hist_with(path, collector.groups_of_all_users, label_type, skipp, skippable)
            collector.reset(gesture)
            plt.close('all')
            time.sleep(0.1)
        groups = users[username]['lbl_groups_fl']
        if len(groups) == 0:
            skippable.add_skipped_thing(f'{gesture}/{username} (no data)', show_message=True)
            return
        collector.groups_of_all_users += groups
        path = get_path(username, gesture, label_type)
        hist_with(path, collector.groups_of_all_users, label_type, skipp, skippable)
        all_groups = all_groups + groups
    skippable = iutils.Skippable(visualize_channel_gesture_callback)
    return skippable.get_callback(), skippable


def generate_timing_historgrams(users, label_type, ud_helper):
    collector = iutils.AllUsersCollector()
    callback, skippable = generate_histogram_callback(
        users, "all_users", label_type, collector, True)
    skippable.mute()
    ud_helper.iterate_users_of_gestures(callback)
    path = get_path("all_users", collector.current_gesture, label_type)
    hist_with(path, collector.groups_of_all_users, label_type, True, skippable)

    path = get_path("all_users", "all_gestures", label_type)
    hist_with(path, collector.groups_of_all_users, label_type, True, skippable)

    skippable.report()


def generate_all_timing_histograms(users, ud_helper):
    lbl_types = ['automatic', 'manual', 'dynamic', 'static']
    for lt in lbl_types:
        generate_timing_historgrams(users, lt, ud_helper)


def show_valuerange_histograms(usernames: List[str], users: List, column: str, remove_outliers: bool,
                               higher_percentile: float, lower_percentile: float, show_overal: bool):
    all_vals = []
    lines = sutils.values_per_user(usernames, users, column, remove_outliers,
                                   higher_percentile, lower_percentile, True)
    for onebigline, username in lines:
        plt.hist(onebigline)
        plt.show()
        plt.close()
        all_vals += list(onebigline)
    if show_overal:
        all_vals = np.array(all_vals)
        plt.hist(all_vals)
        plt.show()
        plt.close()


def get_valurange_hist_path(username, gesture, column):
    fig_base_path = iutils.img_base_path(username, gesture)
    path = f'{fig_base_path}value_distrubution_of_{column}.png'
    return path


def save_valuerange_histograms(usernames: List[str], users: List, column: str, remove_outliers: bool,
                               higher_percentile: float, lower_percentile: float, show_overal: bool,
                               skippable: iutils.Skippable, skipp: bool):
    all_vals = []
    lines = sutils.values_per_user(usernames, users, column, remove_outliers,
                                   higher_percentile, lower_percentile, True)
    for onebigline, username in lines:
        path = get_valurange_hist_path(username, 'all_values', column)
        if skipp and pathlib.Path(path).exists():
            skippable.add_skipped_thing(f'value distribution of {username}/{column}')
            continue
        plt.hist(onebigline)
        plt.savefig(path)
        plt.close()
        all_vals += list(onebigline)
    if show_overal:
        path = get_valurange_hist_path('all_users', 'all_values', column)
        if skipp and pathlib.Path(path).exists():
            skippable.add_skipped_thing(f'value distribution of all_users/{column}')
            return
        all_vals = np.array(all_vals)
        plt.hist(all_vals)
        plt.savefig(path)
        plt.close()
