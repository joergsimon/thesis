from __future__ import annotations
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
            res.append(("aut", self.automatic, other.automatic))
        if not self.manual.approx(other.manual, tolerance):
            res.append(("man", self.manual, other.manual))
        if not self.dynamic.approx(other.dynamic, tolerance):
            res.append(("dyn", self.dynamic, other.dynamic))
        self.diff_static(other, tolerance, res)
        return res

    def diff_static(self, other, tolerance: timedelta, res):
        if (self.static is None) and (other.static is None):
            return
        elif self.static is None:
            if other.static.delta() < tolerance:
                return
            else:
                res.append(("sta", 0, other.static))
        elif other.static is None:
            if self.static.delta() < tolerance:
                return
            else:
                res.append(("sta", self.static, 0))
        elif not self.static.approx(other.static, tolerance):
            res.append(("sta", self.static, other.static))


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


def recover_group(startdate, labelrow):
    start = labelrow['start_glove']
    end = labelrow['end_glove']
    s = pd.Timestamp(startdate + (start * Constants.dt_t))
    e = pd.Timestamp(startdate + (end * Constants.dt_t))
    return (s, e)


def get_automatic_labels(label_data):
    automatic = label_data[label_data['manual_L_vs_automatic_G'] == 'G']
    return automatic


def get_manual_labels(label_data):
    manual = label_data[label_data['manual_L_vs_automatic_G'] == 'L']
    return manual


def get_dynamic_labels(label_data):
    dynamic = label_data[label_data['aut0_dyn1_static2'] == 1]
    return dynamic


def get_static_labels(label_data):
    static = label_data[label_data['aut0_dyn1_static2'] == 2]
    return static
