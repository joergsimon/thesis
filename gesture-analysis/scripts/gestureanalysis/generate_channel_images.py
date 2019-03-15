from __future__ import annotations
from typing import List, Callable
import tqdm
import time
import pathlib
import matplotlib.pyplot as plt
import pandas as pd
from . import specific_utils as sutils
from . import image_utils as iutils
from . import utils


class Skippable:

    def __init__(self, handler: Callable[[Skippable, str, str, tqdm.tqdm_notebook], None]):
        self.skipped_things = []
        self.handler = handler

    def get_callback(self):
        def callback(username: str, gesture: str, bar: tqdm.tqdm_notebook):
            self.handler(self, username, gesture, bar)
        return callback

    def add_skipped_thing(self, thing: str, show_message: bool):
        self.skipped_things.append(thing)
        if show_message:
            print(f'skip {thing} because a image(s) is/are already there')


class ChannelVisTemplate:

    def __init__(self):
        self.average = None
        self.std = None
        self.current_gesture = None
        self.current_column = None
        self.axis = None
        self.path = None
        self.list_of_dfs = None
        self.label_groups = None

    def print_avarage_signal(self):
        av = self.average.loc[:, self.current_column]
        av.plot(linewidth=3, figsize=(18, 5), ax=self.axis, x_compat=True)
        self.axis.set_title(self.current_gesture + ', channel: ' + self.current_column)
        return av

    def fill_standart_deviation_for_standart_signal(self, av: pd.DataFrame):
        s = self.std.loc[:, self.current_column]
        self.axis.fill_between(av.index, av - s, av + s, color='b', alpha=0.2)
        return s

    def print_individual_signal(self, df: pd.DataFrame):
        df.loc[:, self.current_column].plot(subplots=True, ax=self.axis,
                                            alpha=0.7, linewidth=0.5,
                                            legend=False, x_compat=True)

    def iterate_columns(self, skippable: Skippable):
        for col in self.average.columns:
            p = pathlib.Path(self.path + col + '.png')
            if p.exists():
                skippable.add_skipped_thing(p, show_message=True)
                continue
            yield col

    def visualize_channel_with_average_signal_and_std(self):
        fig, ax = plt.subplots()
        self.axis = ax
        av = self.print_avarage_signal()
        s = self.fill_standart_deviation_for_standart_signal(av)
        self.y_min = av.min() - s.max()
        for inst in self.list_of_dfs:
            self.print_individual_signal(inst)
            self.y_min = min(self.y_min, inst.loc[:, self.current_column].min())
        for lg in self.label_groups:
            iutils.add_line_for_label(lg, 'manual', self.average, self.current_column, self.y_min, ax, 'r')
        plt.savefig(self.path + self.current_column + '.png')
        plt.close(1)

    def visualize_gesture(self, path: str, data_per_gesture_instance: List[pd.DataFrame],
                          label_groups: List[sutils.LabelGroup], skippable: Skippable):
        self.label_groups = label_groups
        self.path = path
        self.average = utils.average_of_frames(data_per_gesture_instance)
        self.std = utils.std_of_frames(self.list_of_dfs, average=self.average)
        for col in self.iterate_columns(skippable):
            self.current_column = col
            self.visualize_channel_with_average_signal_and_std()


def generate_visualize_all_channel_user_gesture_combinations_callback(users: List, skipp: bool):
    def visualize_channel_user_gesture_callback(skippable: Skippable,
                                                username: str, gesture: str,
                                                bar: tqdm.tqdm_notebook):
        template = ChannelVisTemplate()
        path = iutils.generate_img_base_path(username, gesture, bar)
        data, label_groups = sutils.data_for_gesture(users, username, gesture)
        template.label_groups = list(label_groups)
        if len(template.label_groups) == 0:
            skippable.add_skipped_thing(f'{gesture}/{username} (no data)', show_message=True)
            return
        data = sutils.drop_labels_and_unused(data)
        if skipp and iutils.check_skip_all(path, data):
            skippable.add_skipped_thing(username, show_message=True)
            return
        ranges = sutils.get_timeranges_tuple(template.label_groups, 'automatic')
        list_of_dfs = sutils.split_df_by_groups(data, ranges)
        template.visualize_gesture(path, list_of_dfs, label_groups, skippable)
        # cleanup
        plt.close('all')
        time.sleep(0.1)
    skippable = Skippable(visualize_channel_user_gesture_callback)
    return skippable.get_callback()


def generate_visualize_all_channel_gesture_combinations_using_all_users_callback(users: List,
                                                                                 collector: AllUsersCollector,
                                                                                 skipp: bool):
    def visualize_channel_gesture_callback(skippable: Skippable,
                                           username: str, gesture: str,
                                           bar: tqdm.tqdm_notebook):
        if gesture != collector.current_gesture:
            if collector.current_gesture is not None:
                template = ChannelVisTemplate()
                path = iutils.generate_img_base_path('all_users', gesture, bar)
                if skipp and iutils.check_skip_all(path, collector.instances_of_all_users[0]):
                    skippable.add_skipped_thing(gesture)
                    return
                template.visualize_gesture(path, collector.instances_of_all_users,
                                           collector.groups_of_all_users, skippable)
                collector.reset(gesture)
                plt.close('all')
                time.sleep(0.1)
        data, groups = sutils.data_for_gesture(users, username, gesture)
        groups = list(groups)
        collector.groups_of_all_users = collector.groups_of_all_users + groups
        if len(groups) == 0:
            skippable.add_skipped_thing(f'{gesture}/{username} (no data)', show_message=True)
            return
        data = sutils.drop_labels_and_unused(data)
        ranges = sutils.get_timeranges_tuple(groups, 'automatic')
        all_instances = sutils.split_df_by_groups(data, ranges)
        collector.instances_of_all_users = collector.instances_of_all_users + all_instances
    skippable = Skippable(visualize_channel_gesture_callback)
    return skippable.get_callback()


class AllUsersCollector:

    def __init__(self):
        self.instances_of_all_users = []
        self.groups_of_all_users = []
        self.current_gesture = None
        self.reset()

    def reset(self, new_gesture):
        self.instances_of_all_users = []
        self.groups_of_all_users = []
        self.current_gesture = new_gesture
