import numpy as np
import pandas as pd
from .utils.freshrotation import euler_matrix
from .utils.freshrotation import vector_slerp
from .cache_control import has_preprocess_basic_cache
from .utils.header_tools import create_new_header
from .utils.index_management import add_new_idx_to_hand


def preprocess_basic(data,const):
    if has_preprocess_basic_cache(const):
        data = pd.read_pickle(const.preprocessed_data_cache_file)
        const.load_preprocess_updates()
        return data
    else:
        print("preprocess data")
        convert_values(data, const)
        convolution_filter(data, const)
        compute_orientation_indipendent_accel(data, const)
        data.to_pickle(const.preprocessed_data_cache_file)
        const.save_preprocess_updates()
        return data


def convert_values(data, const):
    gyro_offset = np.loadtxt("dataingestion/gyro_offset.txt")
    accel_headers = const.filter_raw_header('accel')
    for header in accel_headers:
        data.loc[:,header] /= const.LSB_PER_G # in g
    gyro_header = const.filter_raw_header('gyro')
    for header in gyro_header:
        header_index = gyro_header.index(header)
        imu_index = int(header_index/3)
        index_in_imu = int(header_index%3)
        drift = gyro_offset[imu_index][index_in_imu]
        data.loc[:, header] -= drift
        data.loc[:, header] /= const.LSB_PER_DEG_PER_SEC


def convolution_filter(data, const):
    n = len(data.index)
    LAs = np.zeros((n, const.number_imus * 3))

    Grav = np.zeros((const.number_imus, 3))
    Grav.T[2] = 1

    for k, line in enumerate(data.values):

        for i in range(const.number_imus):
            dx, dy, dz = line[const.get_triple_idxs('gyro',i)] * const.dt * np.pi / 180
            rot = euler_matrix(dx, dy, dz)
            Grav[i] = rot.T.dot(Grav[i])

            norm = np.linalg.norm(line[const.get_triple_idxs('accel',i)])
            if norm > 0.8 and norm < 1.2:
                scale = 0.02 if i > 100 else 0.7
                Grav[i] = vector_slerp(Grav[i], line[const.get_triple_idxs('accel',i)] / norm, scale)

        LAs[k] = line[np.array(const.raw_indices['accel'])] - Grav.reshape(1,21)

    accel_headers = const.filter_raw_header('accel')
    for header in accel_headers:
        index = accel_headers.index(header)
        old_index = const.raw_headers.index(header)
        new_index = len(data.columns)
        h = create_new_header(header,new_index, "lin_accel")
        data[h] = LAs[:,index]
        if 'lin_accel' not in const.raw_indices:
            const.raw_indices['lin_accel'] = []
        const.raw_indices['lin_accel'].append(new_index)
        const.raw_headers.append(h)
        add_new_idx_to_hand(old_index,new_index,False,const)
    print(const.raw_indices['lin_accel'])
    print(const.raw_headers)


def compute_orientation_indipendent_accel(data,const):
    absolute_something("accel","absolute_froce",data,const)
    absolute_something("lin_accel", "absolute_lin_froce", data, const)
    direction_cosine(data,const)


def absolute_something(type,new_type,data,const):
    for i in range(const.number_imus):
        headers = const.get_triples(type,i)
        sqsum = data[headers[0]]**2 + data[headers[1]]**2 + data[headers[2]]**2
        header0 = headers[0]
        old_index = const.raw_headers.index(header0)
        new_index = len(data.columns)
        header0 = header0.replace("_X", "")
        h = create_new_header(header0, new_index, new_type)
        data[h] = np.sqrt(sqsum)
        const.raw_headers.append(h)
        if new_type not in const.raw_indices:
            const.raw_indices[new_type] = []
        const.raw_indices[new_type].append(new_index)
        add_new_idx_to_hand(old_index,new_index,True,const)


def direction_cosine(data,const):
    for i in range(const.number_imus):
        headers = const.get_triples("gyro",i)
        sqsum = data[headers[0]]**2 + data[headers[1]]**2 + data[headers[2]]**2
        rot_force = np.sqrt(sqsum)
        for header in headers:
            old_index = const.raw_headers.index(header)
            new_index = len(data.columns)
            h = create_new_header(header, new_index, "direction_cosine")
            data[h] = data[header] / rot_force
            const.raw_headers.append(h)
            if "direction_cosine" not in const.raw_indices:
                const.raw_indices["direction_cosine"] = []
            const.raw_indices["direction_cosine"].append(new_index)
            add_new_idx_to_hand(old_index, new_index, True, const)


