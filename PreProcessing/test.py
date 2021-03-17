from os.path import sep

from Utils.ReadData import get_file
from biosppy.signals import eeg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm as cm

p = ".." + sep + ".." + sep + "Participants" + sep + "OneMin" + sep
t = "video"
for name, data in get_file(p, t):
    data = data.iloc[:, 1:].to_numpy()
    labels = ['FP1', 'FP2', 'F3', 'FZ', 'F4', 'T7', 'C3', 'CZ', 'C4', 'T8', 'P3', 'PZ', 'P4', 'PO7', 'PO8', 'OZ']
    ts, filtered, ts_feat, theta, alpha_low, alpha_high, beta, gamma, plf_pairs, plf = eeg.eeg(signal=data,
                                                                                               sampling_rate=250,
                                                                                               labels=labels, show=False)
    x = np.corrcoef(plf)
    plt.imshow(x, cmap=cm.get_cmap('rainbow'))
    plt.colorbar()
    plt.show()
