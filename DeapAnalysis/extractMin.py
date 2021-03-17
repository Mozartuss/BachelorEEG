import os
from os.path import sep, exists

import numpy
import pandas as pd

from DeapAnalysis.Utils import yield_data


def delete_leading_zero(num):
    if not num.startswith("0"):
        return num
    else:
        return delete_leading_zero(num[1:])


def convert_deap_to_csv(filename, data, ids, path):
    """
    Convert the deap-matlab-data into trial separated trial csv data and save the file into given destination
    :param filename: the filename of the different files to convert
    :param data: the list data
    :param ids: the csv list headers
    :return: None
    """
    trials_counter = 1
    my_channels = ['FP1', 'FP2', 'F3', 'FZ', 'F4', 'T7', 'C3', 'CZ', 'C4', 'T8', 'P3', 'PZ', 'P4', 'PO7', 'PO8', 'OZ']
    for trials in data:
        trial = []
        counter = 0
        for channels in trials[:32]:
            if ids[counter].upper() in my_channels or ids[counter].upper() == "P8" or ids[counter].upper() == "P7":
                if type(channels) == numpy.float64:
                    channel = [ids[counter], channels]
                else:
                    channel = list(channels[384:])

                trial.append(channel)
            counter += 1
        '''
        save the List vertically
        '''
        data_list = list(map(list, zip(*trial)))

        d = pd.DataFrame(data_list, columns=my_channels)
        p_num = delete_leading_zero(filename.split(".")[0][1:])
        fn = "P" + str(p_num) + "_Trial_" + str(trials_counter)

        if not exists(path + "Participants" + sep + "Participant_" + str(p_num)):
            os.makedirs(path + "Participants" + sep + "Participant_" + str(p_num))

        d.to_csv(path + "Participants" + sep + "Participant_" + str(p_num) + sep + fn + ".csv", index=False)

        trials_counter += 1


if __name__ == '__main__':
    f = ".." + sep + ".." + sep + ".." + sep + "DEAP Dataset" + sep + "Preprocessed Datasets" + sep + "data_preprocessed_python" + sep
    for file, data in yield_data(f):
        convert_deap_to_csv(file, data["data"],
                            ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1',
                             'Oz', 'Pz', 'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz', 'C4', 'T8', 'CP6', 'CP2',
                             'P4', 'P8', 'PO4', 'O2'], f)
