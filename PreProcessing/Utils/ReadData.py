import os
import re
from os.path import exists, sep
import pandas as pd


def natural_sort(in_list):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(in_list, key=alphanum_key)


def get_file(folder, option=""):
    filenames = None

    if option.lower() == "pacman" or option.lower() == "p":
        current_folder = folder + "PacmanTask"
        if exists(current_folder):
            filenames = next(os.walk(current_folder))[2]
            filenames = natural_sort(filenames)
        else:
            print("Path doesn't exist!\nRun SplitEEGData.py first!")
            exit()

    elif option.lower() == "video" or option.lower() == "v":
        current_folder = folder + "VideoTask"
        if exists(current_folder):
            filenames = next(os.walk(current_folder))[2]
            filenames = natural_sort(filenames)
        else:
            print("Path doesn't exist!\nRun SplitEEGData.py first!")
            exit()
    else:
        current_folder = folder
        if exists(current_folder):
            filenames = next(os.walk(current_folder))[2]
            filenames = natural_sort(filenames)

    if len(filenames) >= 1 and current_folder is not None:
        for file in filenames:
            print("Read:", file)
            yield file, pd.read_csv(current_folder + sep + file)
