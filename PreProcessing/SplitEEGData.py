import os
from os.path import sep, exists
import pandas as pd


def split_eeg_in_tasks():
    folder = ".." + sep + ".." + sep + "Participants" + sep
    markers = pd.read_csv(folder + "ActionMarkers.csv", delimiter=";")
    for participant in markers.iterrows():
        p = participant[1]
        name = p["Name"]
        participant = p["Participants"]
        p_start = p["Pacman Start"]
        p_end = p["Pacman End"]
        v_start = p["Video Start"]
        v_end = p["Video End"]

        if pd.isnull(p_start):
            print(name)
        else:
            eeg_comp = pd.read_csv(folder + participant + "_EEG.csv", header=None)

            pacman_table = eeg_comp.loc[eeg_comp[eeg_comp[0].astype(int) == int(p_start)].index[0]:
                                        eeg_comp[eeg_comp[0].astype(int) == int(p_end)].index[-1]]

            video_table = eeg_comp.loc[eeg_comp[eeg_comp[0].astype(int) == int(v_start)].index[0]:
                                       eeg_comp[eeg_comp[0].astype(int) == int(v_end)].index[-1]]

            if not exists(folder + "PacmanTask"):
                os.makedirs(folder + "PacmanTask")

            if not exists(folder + "VideoTask"):
                os.makedirs(folder + "VideoTask")

            pacman_table.to_csv(folder + "PacmanTask" + sep + participant + "_pacman.csv", index=False)
            video_table.to_csv(folder + "VideoTask" + sep + participant + "_video.csv", index=False)


if __name__ == '__main__':
    split_eeg_in_tasks()
