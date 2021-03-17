from os.path import sep
import mne
import numpy as np

from PreProcessing.Utils.ReadData import get_file


def enum(**enums):
    return type('Enum', (), enums)


def visualize(ch, sr, ct, ru, st, a):
    n_channels = len(ch)
    info = mne.create_info(ch_names=ch, sfreq=sr, ch_types=ct)
    #ica = mne.preprocessing.ICA(n_components=16, random_state=97)

    if st:
        b = np.swapaxes(a, 0, 1)
        data = np.divide(b, ru)
    else:
        data = np.divide(a, ru)

    raw = mne.io.RawArray(data, info)
    #ica.fit(raw)
    #ica.apply(raw)
    raw.plot()


if __name__ == '__main__':
    RecordingUnits = enum(NanoVolt=1000000000, MicroVolt=1000000, MilliVolt=1000, Volt=1)
    ShapeType = enum(ChannelTime=False, TimeChannel=True)
    SHAPE_TYPE = ShapeType.TimeChannel
    SAMPLE_RATE = 250
    CHANNELS = ['FP1', 'FP2', 'F3', 'FZ', 'F4', 'T7', 'C3', 'CZ', 'C4', 'T8', 'P3', 'PZ', 'P4', 'PO7', 'PO8', 'OZ']
    CHANNEL_TYPES = "eeg"
    PATH = ".." + sep + ".." + sep + "Participants" + sep + "OneMin" + sep
    # Complete Task
    # f = ".." + sep + ".." + sep + "Participants" + sep
    # One minute sequence of the task
    f = ".." + sep + "Participants" + sep + "OneMin" + sep
    t = "pacman"
    n, data = next(get_file(f, t))
    visualize(CHANNELS, SAMPLE_RATE, CHANNEL_TYPES, RecordingUnits.MicroVolt, SHAPE_TYPE, data.iloc[:, 1:17])
