def add_new_idx_to_hand(base_idx, new_idx, add_to_sensor, const):
    if add_new(base_idx, new_idx, const, 'thumb', 'all'):
        pass
    elif add_new(base_idx, new_idx, const, 'finger_1', 'all'):
        pass
    elif add_new(base_idx, new_idx, const, 'finger_2', 'all'):
        pass
    elif add_new(base_idx, new_idx, const, 'finger_3', 'all'):
        pass
    elif add_new(base_idx, new_idx, const, 'finger_4', 'all'):
        pass
    elif add_new(base_idx, new_idx, const, 'wrist', 'all'):
        if add_new(base_idx, new_idx, const, 'wrist', 'flex'):
            pass
        elif add_new(base_idx, new_idx, const, 'wrist', 'imu'):
            pass
    elif add_new(base_idx, new_idx, const, 'palm', 'all'):
        pass
    else:
        print("(basis) fatal: hand index not found {}".format(base_idx))

    # then add back to sensor:
    if add_to_sensor:
        if add_new(base_idx, new_idx, const, 'flex', 'all'):
            if add_new(base_idx, new_idx, const, 'flex', 'row_1'):
                pass
            elif add_new(base_idx, new_idx, const, 'flex', 'row_2'):
                pass
        elif add_new(base_idx, new_idx, const, 'pressure'):
            pass
        elif add_new(base_idx, new_idx, const, 'accel'):
            pass
        elif add_new(base_idx, new_idx, const, 'gyro'):
            pass
        elif add_new(base_idx, new_idx, const, 'magnetometer'):
            pass
        elif add_new(base_idx, new_idx, const, 'lin_accel'):
            pass
        else:
            print("(basis) fatal: sensor index not found {}".format(new_idx))


def add_new(base_idx, new_idx, const, parent_field, field=None):
    if field is None:
        return add_new_direct(base_idx, new_idx, const, parent_field)
    else:
        return add_new_hierachy(base_idx, new_idx, const, parent_field, field)


def add_new_hierachy(base_idx, new_idx, const, parent_field, field):
    if base_idx in const.raw_indices[parent_field][field]:
        if new_idx not in const.raw_indices[parent_field][field]:
            const.raw_indices[parent_field][field].append(new_idx)
        return True
    return False


def add_new_direct(base_idx, new_idx, const, field):
    if base_idx in const.raw_indices[field]:
        if new_idx not in const.raw_indices[field]:
            const.raw_indices[field].append(new_idx)
        return True
    return False


def add_new_idx_of_feature_to_hand(sensor_idx, feature_idx, const, debug_header, debug_feature):
    # first at to part of hand:
    if add(sensor_idx, feature_idx, const, 'thumb', 'all'):
        pass
    elif add(sensor_idx, feature_idx, const, 'finger_1', 'all'):
        pass
    elif add(sensor_idx, feature_idx, const, 'finger_2', 'all'):
        pass
    elif add(sensor_idx, feature_idx, const, 'finger_3', 'all'):
        pass
    elif add(sensor_idx, feature_idx, const, 'finger_4', 'all'):
        pass
    elif add(sensor_idx, feature_idx, const, 'wrist','all'):
        if add(sensor_idx, feature_idx, const, 'wrist','flex'):
            pass
        elif add(sensor_idx, feature_idx, const, 'wrist','imu'):
            pass
    elif add(sensor_idx, feature_idx, const, 'palm', 'all'):
        pass
    else:
        print("(feature) fatal: hand index not found {} (new: {})".format(sensor_idx, feature_idx))
        print("(feature) fatal: header: {} (feature: {})".format(debug_header, debug_feature))

    # then add back to sensor:
    if add(sensor_idx, feature_idx, const, 'flex','all'):
        if add(sensor_idx, feature_idx, const, 'flex','row_1'):
            pass
        elif add(sensor_idx, feature_idx, const, 'flex','row_2'):
            pass
    elif add(sensor_idx, feature_idx, const, 'pressure'):
        pass
    elif add(sensor_idx, feature_idx, const, 'accel'):
        pass
    elif add(sensor_idx, feature_idx, const, 'gyro'):
        pass
    elif add(sensor_idx, feature_idx, const, 'magnetometer'):
        pass
    elif add(sensor_idx, feature_idx, const, 'lin_accel'):
        pass
    elif add(sensor_idx, feature_idx, const, 'direction_cosine'):
        pass
    elif add(sensor_idx, feature_idx, const, 'absolute_froce'):
        pass
    elif add(sensor_idx, feature_idx, const, 'absolute_lin_froce'):
        pass
    else:
        print("(feature) fatal: sensor index not found {} (new: {})".format(sensor_idx, feature_idx))
        print("(feature) fatal: header: {} (feature: {})".format(debug_header, debug_feature))


def add(sensor_idx, feature_idx, const, parent_field, field=None):
    if field is None:
        return add_direct(sensor_idx, feature_idx, const, parent_field)
    else:
        return add_hierachy(sensor_idx, feature_idx, const, parent_field, field)


def add_hierachy(sensor_idx, feature_idx, const, parent_field, field):
    if sensor_idx in const.raw_indices[parent_field][field]:
        if feature_idx not in const.feature_indices[parent_field][field]:
            const.feature_indices[parent_field][field].append(feature_idx)
        return True
    return False


def add_direct(sensor_idx, feature_idx, const, field):
    if sensor_idx in const.raw_indices[field]:
        if feature_idx not in const.feature_indices[field]:
            const.feature_indices[field].append(feature_idx)
        return True
    return False


def dict_feature_sortet(feature_indexes, const):
    # first at to part of hand:
    dict = const.index_dict()
    for feature_idx in feature_indexes:
        if feature_idx in const.feature_indices['thumb']['all']:
            dict['thumb']['all'].append(feature_idx)
        elif feature_idx in const.feature_indices['finger_1']['all']:
            dict['finger_1']['all'].append(feature_idx)
        elif feature_idx in const.feature_indices['finger_2']['all']:
            dict['finger_2']['all'].append(feature_idx)
        elif feature_idx in const.feature_indices['finger_3']['all']:
            dict['finger_3']['all'].append(feature_idx)
        elif feature_idx in const.feature_indices['finger_4']['all']:
            dict['finger_4']['all'].append(feature_idx)
        elif feature_idx in const.feature_indices['wrist']['all']:
            dict['wrist']['all'].append(feature_idx)
            if feature_idx in const.feature_indices['wrist']['flex']:
                dict['wrist']['flex'].append(feature_idx)
            elif feature_idx in const.feature_indices['wrist']['imu']:
                dict['wrist']['imu'].append(feature_idx)
        elif feature_idx in const.feature_indices['palm']['all']:
            dict['palm']['all'].append(feature_idx)
        else:
            print("(trace) fatal: hand index not found {}".format(feature_idx))
            print("(trace) fatal: header: {}".format(const.feature_headers[feature_idx]))

        # then add back to sensor:
        if feature_idx in const.feature_indices['flex']['all']:
            dict['flex']['all'].append(feature_idx)
            if feature_idx in const.feature_indices['flex']['row_1']:
                dict['flex']['row_1'].append(feature_idx)
            elif feature_idx in const.feature_indices['flex']['row_2']:
                dict['flex']['row_2'].append(feature_idx)
        elif feature_idx in const.feature_indices['pressure']:
            dict['pressure'].append(feature_idx)
        elif feature_idx in const.feature_indices['accel']:
            dict['accel'].append(feature_idx)
        elif feature_idx in const.feature_indices['gyro']:
            dict['gyro'].append(feature_idx)
        elif feature_idx in const.feature_indices['magnetometer']:
            dict['magnetometer'].append(feature_idx)
        elif feature_idx in const.feature_indices['lin_accel']:
            dict['lin_accel'].append(feature_idx)
        else:
            print("(trace) fatal: sensor index not found {}".format(feature_idx))
            print("(trace) fatal: header: {}".format(const.feature_headers[feature_idx]))

    return dict
