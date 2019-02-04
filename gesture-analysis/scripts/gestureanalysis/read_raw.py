import glob
import os.path
import pandas as pd
import tqdm
from .constants import Constants

const = Constants()


def read_raw(base_path, users={}):
    """read raw user data into a dict of form
    {usrname: {
        files : [ ... ], filecount: 3, labels: [ ... ], myo: [ ... ], glove: [ ... ] },
     username2: ... }

        Keyword arguments:
        base_path -- path to a folder containing raw data of users
        users -- optionally already partially filled users dictionary
    """
    files = glob.glob(base_path + "/**/*.csv")
    files.sort()
    for f in tqdm.tqdm(files):
        fname = os.path.basename(f)
        splitted = fname.split("_")
        usr = splitted[0]
        ftype = splitted[1]
        if usr in users:
            if fname in users[usr]["files"]:
                continue
            else:
                users[usr]["files"].append(fname)
            users[usr]["filecount"] = users[usr]["filecount"] + 1
        else:
            users[usr] = {}
            users[usr]["filecount"] = 1
            users[usr]["files"] = [fname]

        names = None
        if ftype == "glove":
            names = const.raw_headers
        if ftype == "label":
            names = const.raw_label_headers

        csvdata = pd.read_csv(f, low_memory=False, header=None, names=names)

        if ftype in users[usr]:
            users[usr][ftype].append({'file' : fname, 'data' : csvdata})
        else:
            users[usr][ftype] = [{'file' : fname, 'data' : csvdata}]
    return users
