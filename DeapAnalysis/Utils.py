import glob
import os
import pickle
import re
from os.path import exists, sep, basename
import pandas as pd


def natural_sort(in_list):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(in_list, key=alphanum_key)


def yield_data(folder, deap=True):
    filenames = None
    folders = None
    if exists(folder):
        folders, filenames = next(os.walk(folder))[1:]
    else:
        print("Folder dosn't exist")
        exit(0)
    filenames = natural_sort(filenames)
    folders = natural_sort(folders)

    if deap:
        if len(filenames) >= 1:
            for file in filenames:
                if file.endswith('.dat'):
                    with open((folder + sep + file), 'rb') as f:
                        print('Load ' + str(file))
                        yield file, pickle.load(f, encoding="latin1")
                elif file.endswith('.csv'):
                    print('Load ' + str(file))
                    yield file, pd.read_csv(file)
    else:
        for participant in folders:
            p = participant.split("_")[-1]

            trials = natural_sort(
                [path for path in glob.glob(folder + sep + participant + sep + 'P' + p + '_Trial_*.csv')
                 if basename(path).startswith('P' + p + '_Trial_')])
            for trial in trials:
                print("Load " + basename(trial))
                yield trial, pd.read_csv(trial)
