from __future__ import annotations
from typing import List, Callable
import pandas as pd
import numpy as np
from datetime import timedelta
from dataclasses import dataclass
from .utils import *
from .constants import Constants


def t_to_t(tpl):
    if tpl is None:
        return None
    start,end = tpl
    return Timerange(start, end)


def tim_to_tu(r):
    if r is None:
        return None
    return (r.start, r.end)


@dataclass
class Timerange(object):
    start: pd.Timestamp
    end: pd.Timestamp

    def approx(self, other: Timerange, tolerance: timedelta):
        start_ok = abs(other.start - self.start) < tolerance
        end_ok = abs(other.end - self.end) < tolerance
        return start_ok and end_ok

    def delta(self):
        return self.end - self.start

    def diff(self, other: Timerange, tolerance: timedelta):
        start_ok = abs(other.start - self.start) < tolerance
        end_ok = abs(other.end - self.end) < tolerance
        if start_ok and end_ok:
            return None
        elif start_ok:
            return ('end diff', self.end, other.end)
        elif end_ok:
            return ('start diff', self.start, other.start)
        else:
            return ('total diff', self, other)



@dataclass
class LabelGroup(object):
    label_name: str
    automatic: Timerange
    manual: Timerange
    static: Timerange
    dynamic: Timerange

    def approx(self, other: LabelGroup, tolerance: timedelta):
        label_ok = other.label_name == self.label_name
        aut_ok = self.automatic.approx(other.automatic, tolerance)
        man_ok = self.manual.approx(other.manual, tolerance)
        dyn_ok = self.dynamic.approx(other.dynamic, tolerance)
        sta_ok = self.check_static(other, tolerance)
        return label_ok and aut_ok and man_ok and dyn_ok and sta_ok

    def check_static(self, other, tolerance: timedelta):
        if (self.static is None) and (other.static is None):
            sta_ok = True
        elif self.static is None:
            if other.static.delta() < tolerance:
                sta_ok = True
            else:
                sta_ok = False
        elif other.static is None:
            if self.static.delta() < tolerance:
                sta_ok = True
            else:
                sta_ok = False
        else:
            sta_ok = self.static.approx(other.static, tolerance)
        return sta_ok

    def diff(self, other: LabelGroup, tolerance: timedelta):
        res = []
        if self.label_name != other.label_name:
            res.append(("name", self.label_name, other.label_name))
        if not self.automatic.approx(other.automatic, tolerance):
            res.append(("aut", self.automatic.diff(other.automatic, tolerance)))
        if not self.manual.approx(other.manual, tolerance):
            res.append(("man", self.manual.diff(other.manual, tolerance)))
        if not self.dynamic.approx(other.dynamic, tolerance):
            res.append(("dyn", self.dynamic.diff(other.dynamic, tolerance)))
        self.diff_static(other, tolerance, res)
        return res

    def diff_static(self, other, tolerance: timedelta, res):
        if (self.static is None) and (other.static is None):
            return
        elif self.static is None:
            if other.static.delta() < tolerance:
                return
            else:
                res.append(("sta", 0, other.static.delta()))
        elif other.static is None:
            if self.static.delta() < tolerance:
                return
            else:
                res.append(("sta", self.static.delta(), 0))
        elif not self.static.approx(other.static, tolerance):
            res.append(("sta", self.static.diff(other.static, tolerance)))

    def verify(self, info: List[str], dyn_static_gap_tolerance: timedelta) -> bool:
        no_error = True
        if self.manual.start < self.automatic.start:
            info.append('manual before automatic')
            no_error = False
        if self.manual.start != self.dynamic.start:
            info.append('manual and dynamic w. different start')
            no_error = False
        if self.static is not None:
            if abs(self.static.start - self.dynamic.end) >= dyn_static_gap_tolerance:
                info.append('dynamic and static gap too large')
                if self.dynamic.end > self.static.end:
                    info.append('dynamic much longer than static')
                if self.static.start < self.dynamic.start:
                    info.append('static much earlier than dynamic')
                no_error = False
            if self.static.end != self.manual.end:
                info.append('static and manual w. different end')
                no_error = False
        elif self.manual.end > self.dynamic.end:
            info.append('dynamic and manual w. different end / static = None')
        if self.manual.end > self.automatic.end:
            info.append('manual takes longer than automatic!')
            no_error = False
        if self.automatic.delta() > timedelta(seconds=3.1):
            info.append('automatic label too long')
            no_error = False
        if self.automatic.delta() < timedelta(seconds=2.9):
            info.append('automatic label too short')
            no_error = False
        return no_error


def get_timeranges(label_groups: List[LabelGroup], timerange_key: str) -> List[Timerange]:
    return list(map(lambda x: getattr(x, timerange_key), label_groups))


def get_timeranges_tuple(label_groups: List[LabelGroup], timerange_key: str) -> List[(pd.Timestamp, pd.Timestamp)]:
    ranges = get_timeranges(label_groups, timerange_key)
    ranges = map(lambda x: tim_to_tu(x), ranges)
    return ranges


def get_timedeltas(label_groups: List[LabelGroup], timerange_key: str):
    trs = get_timeranges(label_groups, timerange_key)
    trs = filter(None, trs)
    return list(map(lambda x: x.delta(), trs))


def filter_gesture_for_timedeltas(label_groups: List[LabelGroup], gesture: str):
    filtered = filter(lambda x: x.label_name == gesture, label_groups)
    return filtered


zero = timedelta(milliseconds=0)
low = timedelta(milliseconds=1500)
mid = timedelta(milliseconds=2500)
high = timedelta(milliseconds=3500)
highend = timedelta(milliseconds=5000)


def four_pin_hist(label_groups: List[LabelGroup], timerange_key: str):
    deltas = get_timedeltas(label_groups, timerange_key)
    return np.histogram(deltas, [zero, low, mid, high, highend])


def datestr_from_filename(fname):
    fdate = fname[-23:-4]
    return fdate


def drop_labels_and_unused(df):
    cols = ['63_Magnetometer_X_ignore_double', '64_Magnetometer_Y_ignore_double', '65_Magnetometer_Z_ignore_double',
            'label_automatic','label_manual','label_dynamic','label_static']
    return df.copy().drop(cols, axis=1)


def split_df_by_groups(df, groups, delta=None):
    all_instances = []
    for (start, end) in groups:
        if delta is None:
            delta = (end - start)
        instance = df[start:start + delta]
        instance = instance.copy()
        instance.index = pd.Timestamp('20180101') + (np.arange(len(instance.index)) * Constants.dt_t)
        all_instances.append(instance.copy())
    return all_instances


def combine_ranges_contained(reference, start_search, list_of_possible_subs):
    s, e = None, None
    new_range = []
    for idx in range(start_search, len(list_of_possible_subs)):
        g = list_of_possible_subs[idx]
        if is_inside(g, reference):
            if s is None:
                s, e = g
            else:
                e = g[1]
        else:
            if s is not None:
                new_range.append((s, e))
                s, e = None, None
                start_search = idx
                break
    return max(0, start_search-1), new_range


# ====== Specific to work with the label_data files =======


def recover_group(startdate, labelrow) -> (pd.Timestamp, pd.Timestamp):
    start = labelrow['start_glove']
    end = labelrow['end_glove']
    s = pd.Timestamp(startdate + (start * Constants.dt_t))
    e = pd.Timestamp(startdate + (end * Constants.dt_t))
    return s, e


def get_automatic_labels(label_data: pd.DataFrame):
    automatic = label_data[label_data['manual_L_vs_automatic_G'] == 'G']
    return automatic


def get_manual_labels(label_data: pd.DataFrame):
    manual = label_data[label_data['manual_L_vs_automatic_G'] == 'L']
    return manual


def get_dynamic_labels(label_data: pd.DataFrame):
    dynamic = label_data[label_data['aut0_dyn1_static2'] == 1]
    return dynamic


def get_static_labels(label_data: pd.DataFrame):
    static = label_data[label_data['aut0_dyn1_static2'] == 2]
    return static


def data_for_gesture(users: List, username: str, gesture: str) -> (List[pd.DataFrame], List[LabelGroup]):
    glove_merged = users[username]['glove_merged']
    g_lbls = glove_merged[glove_merged['label_automatic'] == gesture]
    label_groups = users[username]['lbl_groups_fl']
    lgs = filter_gesture_for_timedeltas(label_groups, gesture)
    return g_lbls, lgs


def data_for_gesture_timealigned(users: List, username: str, gesture: str) -> (List[pd.DataFrame], List[LabelGroup]):
    glove_merged = users[username]['glove_merged'] # get back all data in this case
    label_groups = users[username]['lbl_groups_fl']
    lgs = filter_gesture_for_timedeltas(label_groups, gesture)
    return glove_merged, lgs


def values_per_user(usernames: List[str], users: List, colum, remove_outliers: bool,
                    higher_percentile: float, lower_percentile: float, use_tqtm=False):
    return values_per_user_for_label_type_and_gesture(
        usernames, users, colum, None, 'all_values', remove_outliers,
        higher_percentile, lower_percentile, use_tqtm)


def values_for_user_for_label_type_and_gesture(username: str, users: List, columes,
                                               label_type: str, gesture: str, remove_outliers: bool,
                                               higher_percentile: float, lower_percentile: float):
    def get_data():
        glove_merged = users[username]['glove_merged']
        if gesture is 'all_values':
            data = glove_merged
        elif gesture == 'zero_class':
            data = glove_merged[glove_merged[label_type].isnull()]
        elif gesture == 'all_labels':
            data = glove_merged[glove_merged[label_type].notnull()]
        else:
            data = glove_merged[glove_merged[label_type] == gesture]
        return data

    data = get_data()
    data = data[columes]
    onebigline = data.values.ravel().copy()
    if remove_outliers:
        remove_higher_outliers_with_percentile(onebigline, higher_percentile)
        remove_lower_outliers_with_percentile(onebigline, lower_percentile)
    return onebigline


def values_per_user_for_label_type_and_gesture(usernames: List[str], users: List, columes,
                                               label_type: str, gesture: str, remove_outliers: bool,
                                               higher_percentile: float, lower_percentile: float,
                                               use_tqtm=False):
    uns = tqdm.tqdm_notebook(usernames) if use_tqtm else usernames
    for username in uns:
        if not 'glove_merged' in users[username]:
            print('skipping user' + username)
            continue
        onebigline = values_for_user_for_label_type_and_gesture(
            username, users, columes, label_type, gesture, remove_outliers,
            higher_percentile, lower_percentile)
        yield onebigline, username


def collect_values(usernames: List[str], users: List, colum: str, remove_outliers: bool,
                   higher_percentile: float, lower_percentile: float,
                   use_callback: bool,
                   user_callback: Callable[[np.array, str], None],
                   use_tqtm=False):
    all_vals = []
    lines = values_per_user(usernames, users, colum, remove_outliers, higher_percentile,
                            lower_percentile, use_tqtm=use_tqtm)
    for l, u in lines:
        all_vals += list(l)
        if use_callback:
            user_callback(l, u)
    return all_vals
