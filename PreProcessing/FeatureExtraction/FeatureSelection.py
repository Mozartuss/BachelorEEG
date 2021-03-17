from os.path import sep

import numpy as np
import pandas as pd
from pymrmre import mrmr_ensemble


def select_features(path, type):
    x_data = pd.read_csv(path + "CompleteFeatures.csv")
    x_data = x_data.iloc[:, 1:]
    y_data = pd.read_csv(path + "CompleteFeaturesLabels.csv")
    bins = np.array([1, 2.25, 5, 7.25, 9])
    y_data = pd.DataFrame(np.subtract(np.digitize(y_data[type].to_numpy(), bins, right=True), 2))
    sol = mrmr_ensemble(features=x_data, targets=y_data, solution_length=30).iloc[0][0]
    x = pd.DataFrame(sol)
    x.to_csv(path + "SelectedFeatures" + type.capitalize() + ".csv", index=False)


if __name__ == '__main__':
    f = ".." + sep + ".." + sep + ".." + sep + ".." + sep + "DEAP Dataset" + sep + "Preprocessed Datasets" + sep + "data_preprocessed_python" + sep + "Participants" + sep + "CompleteFeatures" + sep
    predictions = ['valence', 'arousal', 'dominance']
    for pre in predictions:
        select_features(f, pre)
