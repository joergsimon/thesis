from __future__ import annotations
from typing import List
import pathlib
import time
import tqdm
import matplotlib.pyplot as plt
from . import specific_utils as sutils
from . import image_utils as iutils


def draw_and_save_hist(array):
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
    draw_and_save_hist(list(map(lambda x: x.total_seconds(), deltas)))


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

