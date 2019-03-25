from .constants import Constants as const
import numpy as np
import pandas as pd
import tqdm
from .utils import missing_elements
from datetime import datetime, timedelta


def is_corresponding(glove_file, label_file):
    fdate = glove_file[-23:-4]
    ldate = label_file[-23:-4]
    fusr = glove_file[:4]
    lusr = label_file[:4]
    return fdate == ldate and fusr == lusr


def add_label(glovedata, labelrow, column):
    columnidx = glovedata.columns.get_loc(column)
    gesture = labelrow['gesture']
    start = labelrow['start_glove']
    end = labelrow['end_glove']
    glovedata.iloc[start:(end+1), columnidx] = gesture
    #glovedata.loc[(glovedata.index >= start) & (glovedata.index <= end), column] = gesture


def add_labels(gdata, ldata):
    # add labels:

    gdata['label_automatic'] = np.NaN
    gdata['label_manual'] = np.NaN
    gdata['label_dynamic'] = np.NaN
    gdata['label_static'] = np.NaN

    # prepare annotated Automatic Labels:
    automatic = ldata[ldata['manual_L_vs_automatic_G'] == 'G']
    manual = ldata[ldata['manual_L_vs_automatic_G'] == 'L']
    dynamic = manual[manual['aut0_dyn1_static2'] == 1]
    static = manual[manual['aut0_dyn1_static2'] == 2]

    for _, row in automatic.iterrows():
        add_label(gdata, row, 'label_automatic')
    for _, row in manual.iterrows():
        add_label(gdata, row, 'label_manual')
    for _, row in dynamic.iterrows():
        add_label(gdata, row, 'label_dynamic')
    for _, row in static.iterrows():
        add_label(gdata, row, 'label_static')


def transform_index_to_time(fname, gdata, data):
    fdate = fname[-23:-4]
    startdate = datetime.strptime(fdate, "%Y_%m_%d_%H_%M_%S")

    offsets = gdata.index.values * const.dt_t
    times = startdate + offsets
    tmp = pd.to_datetime(times)
    gdata.index = tmp

    if 'glove_merged' in data:
        old_data = data['glove_merged']
        data['glove_merged'] = old_data.append(gdata)
    else:
        data['glove_merged'] = gdata


def preprocess_raw(users, add_label_func, transform_index_to_time_func):
    for key, data in tqdm.tqdm_notebook(users.items()):
        glove_data = data['glove']
        label_data = data['label']
        for (gd, ld) in zip(glove_data, label_data):
            fname = gd['file']
            lfname = ld['file']
            if not is_corresponding(fname, lfname):  # should always be true
                print('ignore user ', fname[0:4])
                #print(fname, '\n', lfname, '\n', "!!!! WTF, WTF, WTF !!!")
                continue

            gdata = gd['data']
            ldata = ld['data']

            # check if index is ok:
            missing_elements(list(gdata.index.values))

            # add labels:

            add_label_func(gdata, ldata)

            # change index to time:
            transform_index_to_time_func(fname, gdata, data)


def is_wrong_label(duration, calc_duration, tolerance):
    if not ((duration > (calc_duration - tolerance)) and (duration < (calc_duration + tolerance))):
        print("(duration > (calc_duration - tolerance)): ", (duration > (calc_duration - tolerance)))
        print("(duration < (calc_duration + tolerance): ", (duration < (calc_duration + tolerance)))
        print('drop label with high disagreement!')
        print(duration)
        print("calculated time for rows: ", calc_duration)
        return True
    return False


low = timedelta(milliseconds=1500)
mid = timedelta(milliseconds=2500)
high = timedelta(milliseconds=3500)


def collect_stats(duration, calc_duration, pins):
    if duration < low:
        pins[0][0] += duration # duration
        pins[0][1] += 1 # count
        pins[0][2] += calc_duration # calculated duration
    elif duration < mid:
        pins[1][0] += duration
        pins[1][1] += 1
        pins[1][2] += calc_duration
    elif duration < high:
        pins[2][0] += duration
        pins[2][1] += 1
        pins[2][2] += calc_duration
    else:
        pins[3][0] += duration
        pins[3][1] += 1
        pins[3][2] += calc_duration

