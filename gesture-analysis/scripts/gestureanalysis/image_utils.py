from __future__ import annotations
from typing import List, Callable
import tqdm
import pathlib
import random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class UserDataHelper:

    def __init__(self, usernames: List[str], gestures: List[str], columns: List[str]):
        self.usernames = usernames
        self.gestures = gestures
        self.columns = columns

    def iterate_gestures_of_users(self, handler: Callable[[str, str, tqdm.tqdm_notebook], None]):
        userbar = tqdm.tqdm_notebook(self.usernames, desc='users')
        for username in userbar:
            userbar.set_description("Processing %s" % username)
            if username == 'AE30':
                print('skipping user' + username)
                continue
            gesturebar = tqdm.tqdm_notebook(self.gestures, desc='gestures', leave=False)
            for gesture in gesturebar:
                handler(username, gesture, gesturebar)

    def iterate_users_of_gestures(self, handler: Callable[[str, str, tqdm.tqdm_notebook], None]):
        gesturebar = tqdm.tqdm_notebook(self.gestures, desc='gestures', leave=False)
        for gesture in gesturebar:
            gesturebar.set_description("Processing %s" % gesture)
            userbar = tqdm.tqdm_notebook(self.usernames, desc='users')
            for username in userbar:
                if username == 'AE30':
                    print('skipping user' + username)
                    continue
                handler(username, gesture, userbar)

    def gen_random_combination(self, num_users: int, num_gestures: int, num_columns: int) -> (List[str], List[str], List[str]):
        usn = [x for x in self.usernames if x != 'AE30'] # no data for AE30
        usrs = random.sample(range(0, len(usn)), num_users)
        usrs = [usn[i] for i in usrs]
        gstrs = random.sample(range(len(self.gestures)), num_gestures)
        gstrs = [self.gestures[i] for i in gstrs]
        cs = self.columns
        if '64_Magnetometer_Y_ignore_double' in self.columns: # this are erroneous data or labels columns
            cs = cs[:-7]
        cols = random.sample(range(len(cs)), num_columns)
        cols = [cs[i] for i in cols]

        return [usrs, gstrs, cols]

    def display_random_generated_images(self, num_users: int, num_gestures: int, num_columns: int):
        usrs, gestrs, cols = self.gen_random_combination(num_users, num_gestures, num_columns)

        # we create a grid per gesture:
        for g in gestrs:
            f, axarr = plt.subplots(num_users, num_columns, figsize=(17, 10))
            f.suptitle(g, y=0.99)
            for row in range(num_users):
                for cs in range(num_columns):
                    path = f'../figures/raw/{usrs[row]}/{g}/{cols[cs]}.png'
                    if pathlib.Path(path).exists():
                        a = mpimg.imread(path)
                        axarr[row, cs].imshow(a)
                    hide_ticks_in_grid(row, cs, axarr)
            for row in range(num_users):
                axarr[row, 0].set_ylabel(usrs[row], rotation=0, size='large')
            for cs in range(num_columns):
                r = 0
                p = 5
                if num_columns > 4:
                    r = 75
                    p = 115
                axarr[0, cs].set_title(cols[cs], rotation=r, pad=p)
            f.tight_layout(pad=0.4)
            plt.show()


def hide_ticks_in_grid(row, cs, axarr):
    plt.setp(axarr[row, cs].get_xticklabels(), visible=False)
    plt.setp(axarr[row, cs].get_yticklabels(), visible=False)
    axarr[row, cs].tick_params(axis='both', which='both', length=0)
    axarr[row, cs].set_xticklabels([])
    axarr[row, cs].set_yticklabels([])


def img_base_path(username: str, gesture: str) -> str:
    fig_base_path = '../figures/raw/' + username + '/' + gesture + '/'
    pathlib.Path(fig_base_path).mkdir(parents=True, exist_ok=True)
    return fig_base_path


def generate_img_base_path(username: str, gesture: str, bar: tqdm.tqdm_notebook) -> str:
    bar.set_description('creating path for ' + gesture + ' of ' + username)
    return img_base_path(username, gesture)


def img_exist(path: str, col: str) -> bool:
    my_file = pathlib.Path(path+col+'.png')
    return my_file.exists()


def check_skip_all(path: str, data: pd.DataFrame) -> bool:
    all_image_exist = True
    for col in data.columns:
        if not img_exist(path, col):
            all_image_exist = False
            break
    return all_image_exist


def add_line_for_label(lg, timerange_key, df_w_fitting_idx, col, y, axis, color):
    aut = lg.automatic
    tr = getattr(lg, timerange_key)
    start = pd.Timestamp('20180101') + (tr.start - aut.start)
    end = pd.Timestamp('20180101') + (tr.end - aut.start)
    x = df_w_fitting_idx.loc[start:end, col].index
    y = [y] * len(x)
    s = pd.Series(y, index=x)
    lbl = f'{timerange_key}: {tr.delta().total_seconds():.2f}s'
    s.plot(subplots=True, ax=axis, color=color, alpha=0.7, linewidth=2, label=lbl, x_compat=True)


class AllUsersCollector:

    def __init__(self):
        self.instances_of_all_users = []
        self.groups_of_all_users = []
        self.current_gesture = None
        self.reset(None)

    def reset(self, new_gesture):
        self.instances_of_all_users = []
        self.groups_of_all_users = []
        self.current_gesture = new_gesture


class Skippable:

    def __init__(self, handler: Callable[[Skippable, str, str, tqdm.tqdm_notebook], None]):
        self.skipped_things = []
        self.handler = handler
        self._mute = False

    def get_callback(self):
        def callback(username: str, gesture: str, bar: tqdm.tqdm_notebook):
            self.handler(self, username, gesture, bar)
        return callback

    def mute(self):
        self._mute = True

    def unmute(self):
        self._mute = False

    def add_skipped_thing(self, thing: str, show_message: bool):
        self.skipped_things.append(thing)
        if show_message and (not self._mute):
            print(f'skip {thing} because a image(s) is/are already there')

    def report(self):
        if len(self.skipped_things) > 0:
            print(f'skipped #{len(self.skipped_things)} image generations')
