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


