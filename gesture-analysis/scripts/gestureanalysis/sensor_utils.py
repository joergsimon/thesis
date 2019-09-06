import pickle
import numpy as np
from gestureanalysis.constants import Constants
from dataclasses import dataclass


@dataclass
class DataBunch:
    """
    This class is used to have one wrapper type to return when using the load_data function. It combines the common
    X and y training data set, the X and y validation data set, the list of used gestures, and the list of used
    features
    """
    X_train: np.array
    y_train: np.array
    X_val: np.array
    y_val: np.array
    gestures: np.array
    feature_names: np.array


def load_data(filename: str):
    """

    :param filename: path to a pickle for a dataset already prepared for training.
    :return: a instance of DataBunch
    """
    with open(filename, "rb") as pickle_file:
        data = pickle.load(pickle_file)
    data_bunch = DataBunch(data['X'], data['y'], data['Xval'], data['yval'], data['gestures'], data['headers'])
    return data_bunch


def select_sensors_from_bunch(data_bunch: DataBunch, select_sensors, classifiers, fit_classifier, alpha: float = 0.5):
    """
    :param data_bunch: the data to run sensor selection on
    :param select_sensors: the sensor selection function to use with the data
    :param classifiers: a list of classifiers to test at each step
    :param fit_classifier: the function for evaluating the classifiers
    :param alpha: a weight for the regularization with the l1 term
    :return:
    """
    results = []
    db = data_bunch
    select_sensors(alpha, db.X_train, db.y_train, db.X_val, db.y_val, db.feature_names.flatten(),
                    classifiers, fit_classifier, results)
    return results


def two_channels_feature_to_index(feature):
    """
    a very small helper function used when a channel encodes a feature originating from two data channels. Then the
    original data channels are n the form WHATEVER_CHANNEL1NUM_CHANNEL2NUM
    :param feature:
    :return:
    """
    components = feature.split("_")
    num1 = int(components[-2])
    num2 = int(components[-1])
    return num1, num2


def computed_feature_to_index(feature):
    """
    This method mapps a list of features names to indices of their original data channels
    :param feature: header of a feature channel
    :return: index in the original data channel
    """
    const = Constants()
    # these features are only imu or gyro, so start counting from back
    if feature.find('Gyro') != -1:
        #print('found gyro')
        if feature.find('X'):
            idx = -3
        if feature.find('Y'):
            idx = -2
        if feature.find('Z'):
            idx = -1
    elif feature.find('Accel') != -1:
        #print('found accel')
        if feature.find('absolute_froce') != -1:
            idx = -4 # just take any accel
        if feature.find('X'):
            idx = -6
        if feature.find('Y'):
            idx = -5
        if feature.find('Z'):
            idx = -4
    #print('idx is now', idx)
    components = feature.split("_")
    if feature.find('Palm') != -1: # it is at the palm
        return const.raw_indices['palm']['all'][idx]
    elif feature.find('Thumb') != -1:
        return const.raw_indices['thumb']['all'][idx]
    elif feature.find('Wrist') != -1:
        return const.raw_indices['wrist']['imu'][idx]
    elif feature.find('Finger') != -1:
        finger_idx = int(components[3])
        #print('select finger nr ', finger_idx)
        #print('finger has idx: ', const.raw_indices[f'finger_{finger_idx}']['all'])
        #print('selected sensor: ', const.raw_indices[f'finger_{finger_idx}']['all'][idx])
        return const.raw_indices[f'finger_{finger_idx}']['all'][idx]


def sensor_array(features):
    """
    given a list of feature channels used, this method counts per sensor how many feature channels are used. This is
    used as a "proxy" to quantify the imprtance of a sensor in recursive sensor elimination RSE.
    :param features: list of used features
    :return: two things: an array of the raw count for each sensor how many features are used, and a map of the sensor
             to the concrete features for analysis
    """
    all_sensors = np.zeros((63,1))
    feature_to_all_sensors = {num: [] for num in range(63)}
    for f in features:
        idx = f.find("_")
        num = f[:idx]
        num = int(num)
        if num < 65:
            all_sensors[num] = all_sensors[num]+1
            feature_to_all_sensors[num].append(f)
        elif num < 1000:
            num = computed_feature_to_index(f)
            if num == -1:
                continue
            all_sensors[num] = all_sensors[num]+1
            feature_to_all_sensors[num].append(f)
        else:
            num1, num2 = two_channels_feature_to_index(f)
            if num1 == -1:
                continue
            all_sensors[num1] = all_sensors[num1]+1
            feature_to_all_sensors[num1].append(f)
            all_sensors[num2] = all_sensors[num2]+1
            feature_to_all_sensors[num2].append(f)
            #print('feature map:')
            #print(feature_to_all_sensors[0])
    # now we want all Accel+Gyro combined to the correct IMUs:
    sensors = np.zeros((26,1))
    feature_to_sensors = {num:[] for num in range(28)}
    sensors[:18] = all_sensors[:18]
    for num in range(18):
        feature_to_sensors[num] = feature_to_all_sensors[num]
    #print('=========================================')
    #print(feature_to_all_sensors[0])
    #print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    #print(feature_to_sensors[0])
    for idx in range(18, 63):
        imu_idx = ((idx-18)//6)+18
        sensors[imu_idx] += all_sensors[idx]
        feature_to_sensors[imu_idx] +=feature_to_all_sensors[idx]
    return sensors, feature_to_sensors


def weakest_idx(sensors):
    """
    given an array with counts how many features are used per sensor, this method returns the index of the sensor who
    has the lowest count but is still used (feature count larger than 0)
    :param sensors: array with counts of how many features are used for each sensor
    :return: index of sensor with lowest count but still used
    """
    m = 1000000
    idx = 0
    for s in range(len(sensors)):
        c = sensors[s]
        if c > 0:
            if c < m:
                m = c
                idx = s
    return idx, m


def best_classifier(results):
    """
    when evaluating several options of classifiers, the result is returned in a list with the validation score of each
    classifier. This method selects the best classifier from this list
    :param results: list evaluation results and stored classifiers from the fit_classifier function
    :return: best classifier
    """
    best_val = 0
    index = 0
    for idx, r in enumerate(results):
        _, s, _, _, _ = r
        if s > best_val:
            best_val = s
            index = idx
    return results[index]
