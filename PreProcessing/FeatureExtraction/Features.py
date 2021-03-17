import os
from os.path import sep, exists

import mne_features
import nolds
import numpy as np
from scipy import stats
import pandas as pd

from PreProcessing.CleanEEG import bandpower
from PreProcessing.Utils.ReadData import get_file


def mean(data):
    return np.mean(data)


def standard_deviation(data):
    return np.std(data)


def afd(data):
    return np.mean(np.absolute(np.diff(data)))


def norma_afd(data):
    return np.mean(stats.zscore(np.absolute(np.diff(data))))


def asd(data):
    a = np.diff(data)
    return np.mean(np.absolute(np.diff(a)))


def norma_asd(data):
    a = np.diff(data)
    return np.mean(stats.zscore(np.absolute(np.diff(a))))


def skewness(data):
    return stats.skew(data)


def kurtosis(data):
    return stats.kurtosis(data)


def hjorth(data):
    first_deriv = np.diff(data)
    second_deriv = np.diff(data, 2)

    var_zero = np.mean(data ** 2)
    var_d1 = np.mean(first_deriv ** 2)
    var_d2 = np.mean(second_deriv ** 2)

    activity = var_zero
    morbidity = np.sqrt(var_d1 / var_zero)
    complexity = np.sqrt(var_d2 / var_d1) / morbidity
    return activity, morbidity, complexity


def petrosian_fractal_dimension(data):
    diff = np.diff(data)
    # x[i] * x[i-1] for i in t0 -> tmax
    prod = diff[1:-1] * diff[0:-2]

    # Number of sign changes in derivative of the signal
    n_delta = np.sum(prod < 0)
    n = len(data)

    return np.log(n) / (np.log(n) + np.log(n / (n + 0.4 * n_delta)))


def get_features(file_data, sample_rate):
    par = []
    bands = {"Theta": [4, 8], "LowerAlpha": [8, 10], "UpperAlpha": [10, 13], "Beta": [13, 30], "Gamma": [30, 45]}
    for channel in file_data:
        data = file_data[channel].to_numpy()
        activity, morbidity, complexity = hjorth(data)
        par.extend((mean(data),
                    standard_deviation(data),
                    afd(data),
                    norma_afd(data),
                    asd(data),
                    norma_asd(data),
                    mne_features.univariate.compute_variance(data),
                    mne_features.univariate.compute_ptp_amp(data),
                    skewness(data),
                    kurtosis(data),
                    bandpower(data, sample_rate, bands["Theta"]),
                    bandpower(data, sample_rate, bands["LowerAlpha"]),
                    bandpower(data, sample_rate, bands["UpperAlpha"]),
                    bandpower(data, sample_rate, bands["Beta"]),
                    bandpower(data, sample_rate, bands["Gamma"]),
                    mne_features.univariate.compute_line_length(data),
                    activity,
                    morbidity,
                    complexity,
                    petrosian_fractal_dimension(data),
                    nolds.hurst_rs(data)))
    return par


if __name__ == '__main__':
    p = ".." + sep + ".." + sep + ".." + sep + "Participants" + sep + "OneMin" + sep + "Filtered" + sep
    t = "pacman"
    features = ["Mean", "Std", "Afd", "Norm_Afd", "Asd", "Norm_Asd", "Variance", "Peak-to-peak", "Skewness", "Kurtosis",
                "Aprox_BP_Theta", "Aprox_BP_LowerAlpha", "Aprox_BP_UpperAlpha", "Aprox_BP_Beta", "Aprox_BP_Gamma",
                "Linelength", "Hjorth_Activity", "Hjorth_Morbidity", "Hjorth_Complexity", "Petrosian", "Hurst"]
    channels = ['FP1', 'FP2', 'F3', 'FZ', 'F4', 'T7', 'C3', 'CZ', 'C4', 'T8', 'P3', 'PZ', 'P4', 'PO7', 'PO8', 'OZ']

    col = []
    for channel in channels:
        for feature in features:
            col.append(channel + "_" + feature)

    participants = []
    index = []
    for name, data in (get_file(p, t)):
        index.append(name.split(".")[0])
        data = data.iloc[:, 1:]
        par = get_features(data, 250)
        participants.append(par)

    print("Finish Pacman")

    t = "video"
    for name, data in (get_file(p, t)):
        index.append(name.split(".")[0])
        data = data.iloc[:, 1:]
        par = get_features(data, 250)
        participants.append(par)

    print("Finish Video")
    d = pd.DataFrame(np.array(participants), index=index, columns=col)

    if not exists(p + "CompleteFeatures"):
        os.makedirs(p + "CompleteFeatures")

    d.to_csv(p + "CompleteFeatures" + sep + "CompleteFeatures.csv", index=True)

    """
        data_s = data.iloc[:, 1:]["FP1"].to_numpy()
        print("Mean:", mean(data_s))
        print("Std:", standard_deviation(data_s))
        print("Afd:", afd(data_s))
        print("Norm_Afd:", norma_afd(data_s))
        print("Asd:", asd(data_s))
        print("Norm_Asd:", norma_asd(data_s))
        print("Skewness:", skewness(data_s))
        print("Kurtosis:", kurtosis(data_s))
        print("Bandpower:", bandpower(data_s, 250, [4, 60]))
        print("Hjorth:", hjorth(data_s))
        print("Petrosian fractal dimension:", petrosian_fractal_dimension(data_s))
        print("Hurst fractal dimension:", nolds.hurst_rs(data_s, nvals=None, fit="RANSAC", debug_plot=False, plot_file=None))
        print("Shannon:", estimate_shannon_entropy(data_s))
    """
