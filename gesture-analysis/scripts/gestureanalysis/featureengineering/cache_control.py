import os
import os.path as path
import sys
from ..constants import Constants


def has_preprocess_basic_cache(const):
    ok = path.isfile(const.preprocessed_data_cache_file)
    ok = ok and path.isfile(const.preprocessed_data_meta)
    return ok


def has_window_cache(const):
    ok = path.isfile(const.window_data_meta)
    ok = ok and path.isfile(const.window_data_cache_file)
    ok = ok and path.isfile(const.window_label_cache_file)
    return ok


def clear_all_cache(const):
    chaches = ["step1","step2","step3"]
    clear_cache(chaches,const)


def clear_cache(cache_steps_to_clear, const):
    for step in cache_steps_to_clear:
        if step == "step1":
            if path.isfile(const.init_data_cache_file):
                os.remove(const.init_data_cache_file)
                os.remove(const.init_data_meta)
        elif step == "step2":
            if path.isfile(const.preprocessed_data_cache_file):
                os.remove(const.preprocessed_data_cache_file)
            if path.isfile(const.preprocessed_data_meta):
                os.remove(const.preprocessed_data_meta)
        elif step == "step3":
            if path.isfile(const.window_data_meta):
                os.remove(const.window_data_meta)
            if path.isfile(const.window_label_cache_file):
                os.remove(const.window_label_cache_file)
            if path.isfile(const.window_data_cache_file):
                os.remove(const.window_data_cache_file)


if __name__ == '__main__':
    const = Constants()
    if sys.argv[1] == 'clear':
        if sys.argv[2] == 'all':
            clear_all_cache(const)
        else:
            clear_cache(sys.argv[2:],const)