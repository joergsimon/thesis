import tqdm
import numpy as np
from functools import reduce
import pickle


def try_pickl_or_recreate(path, recreate_func):
    try:
        with open(path, "rb") as pickle_file:
            res = pickle.load(pickle_file)
    except FileNotFoundError:
        print(f"no pickle for path {path}, have to create first...")
    if res is None:
        res = recreate_func()
        with open(path, "wb") as pickle_file:
            pickle.dump(path, pickle_file)
    return res


def missing_elements(lst):
    start, end = lst[0], lst[-1]
    return sorted(set(range(start, end + 1)).difference(lst))


def find_consecutive_groups(lst, delta, use_tqdm=False):
    groups = []
    start = None
    end = None
    if use_tqdm:
        r = tqdm.tqdm(range(0, len(lst) - 1))
    else:
        r = range(0, len(lst) - 1)
    for idx in r:
        r1 = lst[idx]
        r2 = lst[idx + 1]
        if (r2 - r1) <= delta:  # dt*2 to allow a dropt frame from time to time....?
            if start is None:
                end = None
                start = r1
        else:
            if end is None:
                end = r1
                groups.append((start, end))
                start = None
    if start is not None:
        if end is None:
            end = r1
            groups.append((start, end))
    return groups


def average_of_frames(list_of_dfs):
    average = reduce((lambda x, y: x.add(y, fill_value=0)), list_of_dfs)
    average = average / len(list_of_dfs)
    return average


def std_of_frames(list_of_dfs, average = None):
    if average is None:
        average = average_of_frames(list_of_dfs)
    std = map((lambda x: (x - average) ** 2), list_of_dfs)
    std = reduce((lambda x, y: x.add(y, fill_value=0)), std)
    std = np.sqrt(std / len(list_of_dfs))
    return std


def cmp_t(t1, t2):
    if t1 == t2:
        return True
    else:
        return False


def tuple_in_list(tpl, lst):
    for t in lst:
        if cmp_t(tpl, t):
            return True
    return False


def rest(l1, l2):
    if len(l1) == len(l2):
        return None
    candidate = l1 if len(l1) > len(l2) else l2
    start = min(len(l1), len(l2))
    return candidate[start:]


def is_inside(startend1, startend2):
    s1,e1 = startend1
    s2,e2 = startend2
    if (s1 >= s2) and (s1 <= e2):
        if (e1 <= e2) and (e1 >= s2):
            return True
    return False


def overlaps(startend1, startend2):
    s1,e1 = startend1
    s2,e2 = startend2
    if (s1 >= s2) and (s1 <= e2):
        return True
    elif (e1 <= e2) and (e1 >= s2):
        return True
    else:
        return False


def remove_outliers_with_percentile(vector, percentile, get_idx, next_ok):
    perc = np.percentile(vector, percentile)
    o3kidx = get_idx(vector, perc)
    for i in o3kidx:
        if len(vector) < i+1:
            if next_ok(vector, i, perc):
                vector[i] = (vector[i-1] + vector[i+1])/2
            else:
                vector[i] = vector[i-1]
        else:
            vector[i] = vector[i-1]


def remove_lower_outliers_with_percentile(vector, percentile):

    def get_idx(v, p):
        return np.where(v < p)[0]

    def next_ok(v, i, p):
        return v[i+1] > p

    remove_outliers_with_percentile(vector, percentile, get_idx, next_ok)


def remove_higher_outliers_with_percentile(vector, percentile):

    def get_idx(v, p):
        return np.where(v > p)[0]

    def next_ok(v, i, p):
        return v[i+1] < p

    remove_outliers_with_percentile(vector, percentile, get_idx, next_ok)
