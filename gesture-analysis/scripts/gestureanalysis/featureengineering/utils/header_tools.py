from . import misc_helper as mh
from .index_management import add_new_idx_of_feature_to_hand


def create_new_header(base_header, channel_num, suffix):
    raw_header = strip_start_num(base_header)
    header = "{}_{}_{}".format(channel_num, raw_header, suffix)
    return header


def strip_start_num(header):
    idx = header.find("_")
    return header[idx+1:]


def stat_describe_feature_names():

    sts = ["mean","std","min","25q","median",
            "75q","max","range","var","skew",
            "kurtosis","mode"]
    fsts = list(map(lambda x: f"fft_{x}", sts))
    fftf = ["spectral_centroid", "spectral_entropy","ff1","ff2","ff3",
            "ff4","ff5","freq_5sum","bandwith"]
    cwtsms = list(map(lambda x: f"cwt_sums_{x}", range(0,10)))
    cwts = list(map(lambda x: f"cwt_{x}", sts))
    peaks = ["num_peaks", "peak_min", "peak_max", "peak_mean"]

    return sts + fsts + fftf + cwtsms + cwts + peaks


def tuple_feature_names():
    return ["angle","corr","pval","fftAngle","fftCorr",
            "fftPval", "diff", "diffFFT", "xcorr"]


def create_headers(const):
    create_feature_id_struct(const)
    feature_headers = []
    feature_names = stat_describe_feature_names()
    for header in const.raw_headers:
        if header.startswith('label') or header == 'gesture':
            continue
        sensor_idx = const.raw_headers.index(header)
        idx = header.find("_")
        num = header[:idx]
        offset = len(feature_headers)
        for f_name in feature_names:
            f_idx = feature_names.index(f_name)
            comb_num = "{}_{}".format(num, f_idx)
            h = create_new_header(header, comb_num, f_name)
            feature_headers.append(h)
            #add_new_idx_of_feature_to_hand(sensor_idx, (offset + f_idx), const, header, h)

    finger_flex_line1_tuples = mh.get_combinations(const.raw_indices['flex']['row_1'])
    offset = len(feature_headers)
    feature_names = tuple_feature_names()
    for flex_tuple in finger_flex_line1_tuples:
        for f_name in feature_names:
            f_idx = feature_names.index(f_name)
            header = "{}_{}_{}_{}".format((offset+f_idx), f_name, flex_tuple[0],flex_tuple[1])
            feature_headers.append(header)
            # TODO: ok, that weights one correlational feature double the overal weight. But for counts
            # that might be ok. On the other hand, if we change to a other measure, maybe we need to cound
            # one halve on each sensor...
            add_new_idx_of_feature_to_hand(flex_tuple[0], (offset + f_idx), const, "flex1", header)
            add_new_idx_of_feature_to_hand(flex_tuple[1], (offset + f_idx), const, "flex1", header)
        offset = len(feature_headers)

    finger_flex_line1_tuples = mh.get_combinations(const.raw_indices['flex']['row_2'])
    offset = len(feature_headers)
    feature_names = tuple_feature_names()
    for flex_tuple in finger_flex_line1_tuples:
        for f_name in feature_names:
            f_idx = feature_names.index(f_name)
            header = "{}_{}_{}_{}".format((offset+f_idx), f_name, flex_tuple[0],flex_tuple[1])
            feature_headers.append(header)
            # TODO: ok, that weights one correlational feature double the overal weight. But for counts
            # that might be ok. On the other hand, if we change to a other measure, maybe we need to cound
            # one halve on each sensor...
            add_new_idx_of_feature_to_hand(flex_tuple[0], (offset + f_idx), const, "flex2", header)
            add_new_idx_of_feature_to_hand(flex_tuple[1], (offset + f_idx), const, "flex2", header)
        offset = len(feature_headers)

    const.feature_description['feature_headers_array'] = feature_headers
    const.feature_headers = feature_headers


def create_feature_id_struct(const):
    feature_indices = const.index_dict()
    const.feature_description["indices"] = feature_indices
    const.feature_indices = feature_indices