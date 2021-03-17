import os
from os import sep
from os.path import basename, exists
import pandas as pd
import numpy as np

from DeapAnalysis.Utils import yield_data
from PreProcessing.FeatureExtraction.Features import get_features

if __name__ == '__main__':
    features = ["Mean", "Std", "Afd", "Norm_Afd", "Asd", "Norm_Asd", "Variance", "Peak-to-peak", "Skewness", "Kurtosis",
                "Aprox_BP_Theta", "Aprox_BP_LowerAlpha", "Aprox_BP_UpperAlpha", "Aprox_BP_Beta", "Aprox_BP_Gamma",
                "Linelength", "Hjorth_Activity", "Hjorth_Morbidity", "Hjorth_Complexity", "Petrosian", "Hurst"]
    channels = ["FP1", "FP2", "F3", "FZ", "F4", "T7", "C3", "CZ", "C4","T8", "P3", "PZ", "P4", "PO7", "PO8", "OZ"]

    col = []
    for channel in channels:
        for feature in features:
            col.append(channel + "_" + feature)

    participants = []
    index = []

    f = ".." + sep + ".." + sep + ".." + sep + "DEAP Dataset" + sep + "Preprocessed Datasets" + sep + "data_preprocessed_python" + sep + "Participants" + sep
    for file, data in yield_data(f, False):
        index.append(basename(file).split(".")[0])
        participants.append(get_features(data, 128))

    d = pd.DataFrame(np.array(participants), index=index, columns=col)

    if not exists(f + "CompleteFeatures"):
        os.makedirs(f + "CompleteFeatures")

    d.to_csv(f + "CompleteFeatures" + sep + "CompleteFeatures.csv", index=True)
