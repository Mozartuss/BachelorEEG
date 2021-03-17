import os
from os.path import sep, exists, basename

from DeapAnalysis.Utils import yield_data
import pandas as pd
import numpy as np

from DeapAnalysis.extractMin import delete_leading_zero

labels = []


def convert_labels(filename, data, ids, path):
    """
    Convert the deap-matlab-data into trial separated trial csv data and save the file into given destination
    :param filename: the filename of the different files to convert
    :param data: the list data
    :param ids: the csv list headers
    :param path: Filepath
    :param complete: Option to generate complete label csv or participant based
    :return: None
    """

    d = pd.DataFrame(data, columns=ids)
    p_num = delete_leading_zero(filename.split(".")[0][1:])
    fn = "P" + str(p_num) + "_Labels"

    if not exists(path + "Participants" + sep + "Participant_" + str(p_num)):
        os.makedirs(path + "Participants" + sep + "Participant_" + str(p_num))

    d.to_csv(path + "Participants" + sep + "Participant_" + str(p_num) + sep + fn + ".csv", index=False)


if __name__ == '__main__':
    f = ".." + sep + ".." + sep + ".." + sep + "DEAP Dataset" + sep + "Preprocessed Datasets" + sep + "data_preprocessed_python" + sep
    # for file, data in yield_data(f):
    #     labels = data["labels"]
    #     convert_labels(file, labels, ["valence", "arousal", "dominance", "liking"], f)

    for file, data in yield_data(f, True):
        for label in data["labels"]:
            labels.append(label)
    d = pd.DataFrame(labels, columns=["valence", "arousal", "dominance", "liking"])

    if not exists(f + "Participants" + sep + "CompleteFeatures"):
        os.makedirs(f + "Participants" + sep + "CompleteFeatures")

    d.to_csv(f + "Participants" + sep + "CompleteFeatures" + sep + "CompleteFeaturesLabels" + ".csv", index=False)
