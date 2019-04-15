import timeit
import sys
import numpy as np
import pandas as pd
import scipy as sc
import scipy.stats
from .utils.header_tools import create_headers
from .cache_control import has_window_cache
from .utils import misc_helper as mh


def get_windows(data,const):
    if has_window_cache(const):
        const.load_window_updates()
        newData = pd.read_pickle(const.window_data_cache_file)
        newLabels = pd.read_pickle(const.window_label_cache_file)
        return (newData, newLabels)
    else:
        newData, newLabels = transform_to_windows(data,const)
        const.save_window_updates()
        newData.to_pickle(const.window_data_cache_file)
        newLabels.to_pickle(const.window_label_cache_file)
        return (newData, newLabels)


def transform_to_windows(data,const):
    create_headers(const)

    print("flex const index trace info / transform_to_windows:")
    print(len(const.feature_indices['flex']['row_1']))
    print(len(const.feature_indices['flex']['row_2']))

    start_time = timeit.default_timer()

    num_windows = (len(data) - const.window_size) / const.step_size

    print('creating empy frame with len {}'.format(num_windows))

    t1 = timeit.default_timer()
    print(data.info())
    data = data.astype('float64')
    t2 = timeit.default_timer()
    print("converting to float took {}s".format(t2 - t1))

    matrix = np.zeros(shape=(num_windows,len(const.feature_headers))) #None
    labelInfo = []

    for i in range(num_windows):
        if i % 20 == 0:
            progress = (i / float(num_windows))
            msg = '\r[{0}] {1:.2f}%'.format('#' * int(progress * 10), progress * 100)
            sys.stdout.write(msg)
        offset = i * const.step_size

        # COMPUTE Features for Single Signals:
        subframe = data.iloc[offset:(offset + const.window_size)]
        subframe = subframe._get_numeric_data()
        mat = single_value_features(subframe.values[:, 0])
        sub_range = range(1, len(subframe.columns))
        for j in sub_range:
            if subframe.columns[j] != 'gesture':
                vec = single_value_features(subframe.values[:, j])
                mat = np.column_stack((mat, vec))
                # do not forget to add peaks!
        mat = np.array(np.ravel(mat))

        corr_mat = None
        finger_flex_line1_tuples = mh.get_combinations(const.raw_indices['flex']['row_1'])
        for idx1,idx2 in finger_flex_line1_tuples:
            vec = correlational_features(subframe.values[:, idx1], subframe.values[:, idx2])
            if corr_mat is None:
                corr_mat = vec
            else:
                corr_mat = np.column_stack((corr_mat, vec))

        finger_flex_line2_tuples = mh.get_combinations(const.raw_indices['flex']['row_2'])
        for idx1, idx2 in finger_flex_line2_tuples:
            vec = correlational_features(subframe.values[:, idx1], subframe.values[:, idx2])
            corr_mat = np.column_stack((corr_mat, vec))

        mat = np.append(mat,np.array(np.ravel(corr_mat)))

        # ok, maybe getting alls tuples is dump, maybe it is better to list all meaningful combinations
        # like all finger rows, or all finger IMUs and so on...

        #flex_data = data.ix[offset:offset + const.window_size, Constants.flex_map]
        #row1 = flex_data.ix[:, Constants.hand_row_1]
        #row2 = flex_data.ix[:, Constants.hand_row_2]
        #res = self.computeTupelFeatures(row1, newHeaders, "flex_row1")
        #mat = np.append(mat, res)
        #res = self.computeTupelFeatures(row2, newHeaders, "flex_row2")
        #mat = np.append(mat, res)

        counts = subframe.groupby("gesture").size()
        labelInfo.append(counts)

        matrix[i,:] = mat
        # if matrix is None:
        #     matrix = mat
        # else:
        #     matrix = np.vstack((matrix, mat))

    newData = pd.DataFrame(matrix, index=range(num_windows), columns=const.feature_headers)
    newLabels = pd.DataFrame(labelInfo)

    print('\nCleaning ...')

    l = pd.isnull(newData).any(1).nonzero()[0]
    print(l)

    print('Done aggregating')

    time = timeit.default_timer() - start_time
    print('exec took {}s'.format(time))

    return (newData, newLabels)

#
# Spectral Entropy Algorithm:
# http://stackoverflow.com/questions/21190482/spectral-entropy-and-spectral-energy-of-a-vector-in-matlab
# alternatives for computing the PSD:
# scipy.signal.welch
# http://docs.scipy.org/doc/scipy-0.16.1/reference/generated/scipy.signal.welch.html#scipy.signal.welch
# matplotlib.mlab.psd
# http://matplotlib.org/api/mlab_api.html#matplotlib.mlab.psd
#
def single_value_features(array):
   resMin = np.nanmin(array)
   resMax = np.nanmax(array)
   resRange = resMax - resMin
   resMedian = np.median(array)
   resMean = np.nanmean(array)
   resStd = np.nanstd(array)
   resVar = np.nanvar(array)
   res25Q = np.percentile(array, 25)
   res75Q = np.percentile(array, 75)
   resSkew = sc.stats.skew(array)
   resKurtosis = sc.stats.kurtosis(array)
   resMode = sc.stats.mode(array, axis=None)[0]
   length = len(array)
   y = np.fft.rfft(array)
   magnitudes = np.abs(y)
   magnitudes = np.delete(magnitudes, 0)
   freqs = np.fft.rfftfreq(length, d=(1./58))
   freqs = freqs[np.where(freqs >= 0)]
   freqs = np.delete(freqs, 0)
   freqs = np.abs(freqs)
   spectral_centroid = np.sum(magnitudes*freqs)/np.sum(magnitudes)
   psd = pow(magnitudes, 2)/freqs
   psdsum = sum(psd)
   psdnorm = psd/psdsum
   spectral_entropy = sc.stats.entropy(psdnorm)
   freq_5sum = freqs[0] + freqs[1] + freqs[2] + freqs[3] + freqs[4];
   bandwith = max(freqs)-min(freqs)

   # return np.array([resMean, resStd, resMin, res25Q, resMedian,
   #                  res75Q, resMax, resRange, resVar, resSkew,
   #                  resKurtosis, resMode, spectral_centroid,
   #                  spectral_entropy, freqs[0], freqs[1], freqs[2],
   #                  freqs[3], freqs[4], freq_5sum, bandwith])

   # IMPORTANT: If you update here, also update the headers in header tools to have the same order!!!!
   return np.array([resMean, resStd, resMin, res25Q, resMedian,
                    res75Q, resMax, resRange, resVar, resSkew,
                    resKurtosis, resMode, spectral_centroid,
                    spectral_entropy, freqs[0], freqs[1], freqs[2],
                    freqs[3], freqs[4], freq_5sum, bandwith])

def correlational_features(array1, array2):
    # compute correlations, vectors, threshholds....
    # signal correlation:
    # http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.signal.correlate.html
    #
    # angle:
    vec1 = array1 / np.linalg.norm(array1)
    vec2 = array2 / np.linalg.norm(array2)
    angle = np.arccos(np.dot(vec1, vec2))
    corr, pval = scipy.stats.spearmanr(array1, array2)
    # inspired from http://svn.gna.org/svn/relax/tags/4.0.0/lib/geometry/vectors.py
    fV1 = np.fft.rfft(array1)
    fV2 = np.fft.rfft(array2)

    i_v1v2 = np.dot(fV1, fV2.conj().T)
    i_v1v1 = np.dot(fV1, fV1.conj().T)
    i_v2v2 = np.dot(fV2, fV2.conj().T)
    ratio = i_v1v2.real / (np.sqrt(i_v1v1).real * np.sqrt(i_v2v2).real)
    fftAngle = np.arccos(ratio)

    fftCorr, fftPval = scipy.stats.spearmanr(fV1, fV2)

    return np.array([angle, corr, pval, fftAngle, fftCorr, fftPval])