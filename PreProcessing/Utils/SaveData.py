import os
from os.path import exists, sep


def save_file(data, task, path, folder, name, index=False):
    """
    :param index:
    :param data: Pandas dataframe
    :param task: pacman/p or video/v
    :param path: the save path
    :param folder: the new folder where to save
    :param name: the data name with .csv
    """
    task_folder = "PacmanTask" if task.lower() == "p" or task.lower() == "pacman" else "VideoTask"

    if not exists(path + folder + sep + task_folder):
        os.makedirs(path + folder + sep + task_folder)

    print("save", name, "in", path + folder + sep + task_folder)
    data.to_csv(path + folder + sep + task_folder + sep + name, index=index)
