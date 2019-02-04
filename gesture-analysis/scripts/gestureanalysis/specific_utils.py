import pandas as pd
import numpy as np
from .constants import Constants


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

