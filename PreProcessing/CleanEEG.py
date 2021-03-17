from os import sep

from scipy.signal import sosfilt
import pandas as pd
import numpy as np
from scipy.integrate import simps
from mne.time_frequency import psd_array_multitaper

from PreProcessing.Plotter import gen_plt
from PreProcessing.Utils.Filter import butter_bandpass
from PreProcessing.Utils.ReadData import get_file
from PreProcessing.Utils.SaveData import save_file


def remove_artefacts(data, columns):
    time = data.iloc[:, 0:1].to_numpy()
    eeg = data.iloc[:, 1:17]
    filtered_eeg = [time]
    sos = butter_bandpass(4, 60, 250)
    for channel in eeg:
        ch_data = eeg[channel].to_numpy()
        y = sosfilt(sos, ch_data)
        filtered_eeg.append(y)

    dataframe = pd.DataFrame(list(map(list, zip(*filtered_eeg))), columns=columns)
    return dataframe


def separate_freqs(data):
    time = data["Timestamp"].tolist()
    time.insert(0, "Timestamp")
    eeg = data.iloc[:, 1:17]
    filtered_eeg = [time]

    for channel in eeg:
        ch_data = eeg[channel].to_numpy()
        ch_name = eeg[channel].name

        theta = list(sosfilt(butter_bandpass(4, 8, 250), ch_data))
        theta.insert(0, str(ch_name) + "_Theta")
        lower_alpha = list(sosfilt(butter_bandpass(8, 10, 250), ch_data))
        lower_alpha.insert(0, str(ch_name) + "_LowerAlpha")
        upper_alpha = list(sosfilt(butter_bandpass(10, 13, 250), ch_data))
        upper_alpha.insert(0, str(ch_name) + "_UpperAlpha")
        beta = list(sosfilt(butter_bandpass(13, 30, 250), ch_data))
        beta.insert(0, str(ch_name) + "_Beta")
        gamma = list(sosfilt(butter_bandpass(30, 60, 250), ch_data))
        gamma.insert(0, str(ch_name) + "_Gamma")
        filtered_eeg.extend((theta, lower_alpha, upper_alpha, beta, gamma))

    f_sol = pd.DataFrame(list(map(list, zip(*filtered_eeg))))
    f_sol.columns = f_sol.iloc[0]
    return f_sol[1:]


def bandpower(data, sample_freq, band):
    band = np.asarray(band)
    low, high = band

    psd, freqs = psd_array_multitaper(data, sample_freq, adaptive=True,
                                      normalization='full', verbose=0)
    freq_res = freqs[1] - freqs[0]

    # Find index of band in frequency vector
    idx_band = np.logical_and(freqs >= low, freqs <= high)

    # Integral approximation of the spectrum using parabola (Simpson's rule)
    bp = simps(psd[idx_band], dx=freq_res)
    return bp


if __name__ == '__main__':
    p = ".." + sep + ".." + sep + "Participants" + sep + "OneMin" + sep + "Filtered" + sep
    t = "pacman"
    columns = ['Timestamp', 'FP1', 'FP2', 'F3', 'FZ', 'F4', 'T7', 'C3', 'CZ', 'C4', 'T8', 'P3', 'PZ', 'P4', 'PO7',
               'PO8', 'OZ']

    for name, data in get_file(p, t):
        # f_data = remove_artefacts(data, columns)
        # save_file(data=f_data, task=t, path=p, folder="Filtered", name=name)
        # gen_plt(data=f_data, title="Filtered_" + name.split(".")[0], save=True, name=name, new_folder="Pictures", path=p + "Filtered" + sep, task=t)
        s_data = separate_freqs(data)
        save_file(data=s_data, task=t, path=p, folder="SplitFreqs", name=name)
        gen_plt(data=s_data, save=True, name=name, new_folder="Pictures", path=p + "SplitFreqs" + sep, task=t,
                columns=list(s_data.iloc[:, 1:].columns))


