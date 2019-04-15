import pandas as pd
import numpy as np
import os
import os.path as path

class DirectoryBasedReader:

    def __init__(self, rootPath, const):
        self.root = rootPath
        self.const = const

    def get_file_tuples_for_dir(self):
        files = [f for f in os.listdir(self.root) if path.isfile(path.join(self.root, f))]
        labels = [l for l in files if "label" in l]
        labels = sorted(labels)
        gl_data = [g for g in files if "glove" in g]
        gl_data = sorted(gl_data)
        tuples = zip(gl_data, labels)
        return tuples

    def read(self, datafname, labelfname, knowngestures):
        datapath = self.root + datafname
        labelpath = self.root + labelfname
        data = pd.read_csv(datapath, sep=',', header=None, names=self.const.raw_headers)
        #data[self.const.gesture_field] = pd.Series(np.zeros(data.size, dtype=np.int))
        data = data.drop(u'63_Magnetometer_X_ignore_double', 1)
        data = data.drop(u'64_Magnetometer_Y_ignore_double', 1)
        data = data.drop(u'65_Magnetometer_Z_ignore_double', 1)
        labels = pd.read_csv(labelpath, sep=',', header=None)
        gestures = pd.Series(np.zeros(data.size, dtype=np.int))
        for index, label in labels.iterrows():
            if label[0] == self.const.label_type_automatic:
                gesture, start, end = label[1], int(label[3]), int(label[4])
                if gesture not in knowngestures:
                    knowngestures.append(gesture)
                    if (end - start > 300):
                        print("large difference found! {}".format(end - start))
                    gesture_num = knowngestures.index(gesture) + 1
                    if (start < gestures.size):
                        real_end = min(end, gestures.size-1)
                        gestures[start:real_end] = gesture_num

        if not 'gesture' in self.const.raw_headers:
            self.const.raw_headers.append('gesture')
        data['gesture'] = gestures

        return data
