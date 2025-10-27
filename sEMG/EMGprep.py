import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.signal import butter, lfilter, filtfilt
import scipy


def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    return butter(order, [lowcut / nyq, highcut / nyq], btype='band', output='sos')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def notch_pass_filter(data, center, interval=30, sr=2000, normalized=False):
    center = center / (sr / 2) if normalized else center
    b, a = scipy.signal.iirnotch(center, center / interval, sr)
    filtered_data = scipy.signal.filtfilt(b, a, data)
    return filtered_data

def window_RMS(a, window_size):
    a2 = np.power(a, 2)
    window = np.ones(window_size) / float(window_size)
    return np.sqrt(np.convolve(a2, window, 'same'))

def median_freq(emg_signal, sampling_rate=2000):
    f, Pxx = scipy.signal.welch(emg_signal, fs=sampling_rate, nperseg=len(emg_signal))
    cumsum_Pxx = np.cumsum(Pxx)
    median_freq_index = np.argmin(np.abs(cumsum_Pxx - cumsum_Pxx[-1] / 2))
    median_frequency = f[median_freq_index]
    return median_frequency

def mean_power_freq(emg_signal, sampling_rate=2000):
    f, Pxx = scipy.signal.welch(emg_signal, fs=sampling_rate, nperseg=len(emg_signal))
    weighted_frequencies = f * Pxx
    mpf = np.sum(weighted_frequencies) / np.sum(Pxx)
    return mpf

def calculate_fApEn(data, m=2, r=0.2, n=2):
    from scipy.spatial.distance import pdist, squareform
    def _phi(m):
        X = np.array([data[i:i + m] for i in range(len(data) - m + 1)])
        dist_matrix = squareform(pdist(X, metric='chebyshev'))
        fuzzy_similarity = np.exp(-np.power(dist_matrix / r, n))
        C = np.sum(fuzzy_similarity, axis=0) / (len(data) - m + 1)
        return np.sum(np.log(C)) / (len(data) - m + 1)
    return _phi(m) - _phi(m + 1)


def process_emg_files(base_dir, categories, num_subjects=10):
    '''
    Process EMG data files across specified categories and extract multiple signal features.
    
    Parameters:
        base_dir (str): Base directory containing EMG data folders.
        categories (list of str): List of category or session folder names, e.g. ['01_PRE', '02_POST', '03_FOLLOW'].
        num_subjects (int, optional): Number of subjects per category to process. Default is 10.

    Returns:
        Saves CSV files for each extracted feature in the base directory.

    '''
    
    data_dir = []
    for category in categories:
        data_dir.append(glob.glob(os.path.join(base_dir, category, '*', '*.xlsx')))

    max_val = [[{} for _ in range(num_subjects)] for _ in range(len(categories))]
    mean_val = [[{} for _ in range(num_subjects)] for _ in range(len(categories))]
    std_val = [[{} for _ in range(num_subjects)] for _ in range(len(categories))]
    median_val = [[{} for _ in range(num_subjects)] for _ in range(len(categories))]
    area_val = [[{} for _ in range(num_subjects)] for _ in range(len(categories))]
    quartile3_val = [[{} for _ in range(num_subjects)] for _ in range(len(categories))]
    MDF = [[{} for _ in range(num_subjects)] for _ in range(len(categories))]
    MPF = [[{} for _ in range(num_subjects)] for _ in range(len(categories))]
    fApEn = [[{} for _ in range(num_subjects)] for _ in range(len(categories))]

    for k, dirs in enumerate(data_dir):
        for dir_path in tqdm(dirs):
            a = os.path.normpath(dir_path).split(os.sep)[-2:]
            b = os.path.basename(dir_path).split('_')[-4:-2]
            data_name = f"{a[0][3:]}_{a[1]} {b[0]}_{b[1]}"

            data = pd.read_excel(dir_path)
            if 'X[s].1' in data.columns:
                del data['X[s].1']
            if len(data) > 20000:
                data = data.iloc[4000:-4000]
            else:
                data = data.iloc[2000:-2000]
            signal = data[data.columns[1]]

            signal = signal - np.mean(signal)
            bp_data = butter_bandpass_filter(signal, 10, 500, 2000, 2)
            notch_data = notch_pass_filter(bp_data, 60, 10, 2000, True)
            detrended = scipy.signal.detrend(notch_data)
            detrended = detrended - np.mean(detrended)
            rectification_data = np.abs(detrended)[4000:]

            window_size = 250
            staticRMS = window_RMS(rectification_data, window_size)
            staticRMS = pd.DataFrame(staticRMS)

            thre = float(staticRMS.mean() + staticRMS.std())
            tmp = staticRMS[staticRMS[0] >= thre][0]

            if len(tmp.index) == 0:
                thre /= 2
                tmp = staticRMS[staticRMS[0] >= thre][0]

            revised = False
            if tmp.index[0] + 6000 > len(staticRMS):
                revisedRMS = staticRMS.iloc[:tmp.idxmax() - 2000]
                thre = float(revisedRMS.mean() + revisedRMS.std())
                tmp = staticRMS[staticRMS[0] >= thre][0]
                revised = True

            RMS_AUC = np.sum(staticRMS[tmp.index[0]:tmp.index[0] + 6000])
            tmp_desc = pd.DataFrame(staticRMS[tmp.index[0]:tmp.index[0] + 6000].describe())

            text_ = []
            for i in pd.DataFrame(tmp_desc)[0]:
                text_.append('%.2e' % i)
            text_.append('%.2e' % RMS_AUC[0])
            text_ = pd.DataFrame(np.array(text_).reshape(-1))

            max_val[k][int(a[1].split('-')[-1]) - 1][data_name.split('(')[-1][:-1]] = text_[7]
            mean_val[k][int(a[1].split('-')[-1]) - 1][data_name.split('(')[-1][:-1]] = text_[1]
            std_val[k][int(a[1].split('-')[-1]) - 1][data_name.split('(')[-1][:-1]] = text_[2]
            median_val[k][int(a[1].split('-')[-1]) - 1][data_name.split('(')[-1][:-1]] = text_[5]
            area_val[k][int(a[1].split('-')[-1]) - 1][data_name.split('(')[-1][:-1]] = text_[8]
            quartile3_val[k][int(a[1].split('-')[-1]) - 1][data_name.split('(')[-1][:-1]] = text_[6]
            MDF[k][int(a[1].split('-')[-1]) - 1][data_name.split('(')[-1][:-1]] = median_freq(notch_data)
            MPF[k][int(a[1].split('-')[-1]) - 1][data_name.split('(')[-1][:-1]] = mean_power_freq(notch_data)
            fApEn[k][int(a[1].split('-')[-1]) - 1][data_name.split('(')[-1][:-1]] = calculate_fApEn(notch_data)

    vals = [max_val, mean_val, std_val, median_val, area_val, quartile3_val, MDF, MPF, fApEn]
    save_name = ['max_amplitude', 'mean_amplitude', 'std_amplitude', 'median_amplitude', 'AUC_amplitude', '75%_amplitude', 'median_frequency', 'mean_power_frequency', 'fApEn']

    for i, v in enumerate(vals):
        df = pd.concat([pd.DataFrame(v[0]), pd.DataFrame(v[1]), pd.DataFrame(v[2])], axis=1)
        df.to_csv(base_dir + save_name[i] + '.csv', index=False)


