import pandas as pd
import numpy as np
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


@dataclass
class LabelGroup(object):
    label_name: str
    automatic: Timerange
    manual: Timerange
    static: Timerange
    dynamic: Timerange


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
                start = idx
                break
    return max(0, start-1), new_range
