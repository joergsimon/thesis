from typing import List, Callable
import tqdm
import pathlib
import random
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
        usrs = random.sample(range(0, len(self.usernames)), num_users)
        usrs = [self.usernames[i] for i in usrs]
        gstrs = random.sample(range(len(self.gestures)), num_gestures)
        gstrs = [self.gestures[i] for i in gstrs]
        cols = random.sample(range(len(self.columns)), num_columns)
        cols = [self.columns[i] for i in cols]

        return [usrs, gstrs, cols]

    def display_random_generated_images(self, num_users: int, num_gestures: int, num_columns: int):
        usrs, gestrs, cols = self.gen_random_combination(num_users, num_gestures, num_columns)

        # we create a grid per gesture:
        for g in gestrs:
            f, axarr = plt.subplots(num_users, num_columns, figsize=(20, 10))
            for row in range(num_users):
                for cs in range(num_columns):
                    axarr[row, cs].set_title(f'{usrs[row]}/{g}/{cols[cs]}')
                    path = f'../figures/raw/{usrs[row]}/{g}/{cols[cs]}.png'
                    a = mpimg.imread(path)
                    axarr[row, cs].imshow(a)
            plt.show()


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
