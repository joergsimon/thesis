from .reader import DirectoryBasedReader
import os.path as path
import pandas as pd
import pickle


class InitialInput:

    def __init__(self, const):
        self.const = const
        self.reader = DirectoryBasedReader(const.init_data_dir, const)

    def read_all_data_init(self):
        data = None
        file_tuples = self.reader.get_file_tuples_for_dir()
        if self.has_valid_cache(file_tuples):
            return self.read_cache()
        else:
            data = self.read_all_data(file_tuples)
        self.save_cache(file_tuples, data)
        return data

    def has_valid_cache(self, new_file_tuples):
        cache_ok = path.isfile(self.const.init_data_cache_file)
        if not cache_ok:
            return False
        cache_ok = path.isfile(self.const.init_data_meta)
        if not cache_ok:
            return False
        with open(self.const.init_data_meta) as file:
            old_file_tuples = pickle.load(file)
            cache_ok = cache_ok and (old_file_tuples == new_file_tuples)
        return cache_ok

    def read_cache(self):
        data = pd.read_pickle(self.const.init_data_cache_file)
        return data

    def read_all_data(self, file_tuples):
        data = None
        knowngestures = []
        for glove_data, label_data in file_tuples:
            d = self.reader.read(glove_data, label_data, knowngestures)
            if data is None:
                data = d
            else:
                data = pd.concat([data, d], ignore_index=True)
        return data

    def save_cache(self, file_tuples, data):
        with open(self.const.init_data_meta, 'wb') as file:
            pickle.dump(file_tuples, file)
        data.to_pickle(self.const.init_data_cache_file)