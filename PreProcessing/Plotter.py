import os
from os.path import sep, exists
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
from matplotlib.collections import LineCollection

from PreProcessing.Utils.ReadData import get_file


def gen_plt(data, name="", path="", task="", new_folder="", title="", save=False, columns=None):
    if columns is None:
        columns = ['FP1', 'FP2', 'F3', 'FZ', 'F4', 'T7', 'C3', 'CZ', 'C4', 'T8', 'P3', 'PZ', 'P4', 'PO7', 'PO8', 'OZ']

    @ticker.FuncFormatter
    def major_formatter(x, pos):
        return str(int(x) / 250) + "s"

    # Remove Timeline Column
    data = data.iloc[:, 1:].to_numpy()
    n_samples, n_rows = data.shape
    t = np.arange(n_samples)

    # Plot the EEG
    ticktocks = []
    fig = plt.figure(figsize=(80, 30))
    ax = fig.add_subplot(1, 1, 1)
    # Add 250 ticks (1 sec) on every x-axis side to set space
    ax.set_xlim(-250, n_samples + 250)
    plt.xticks(np.arange(0, n_samples + 1, 2500))
    ax.xaxis.set_major_formatter(major_formatter)
    dmin = data.min()
    dmax = data.max()
    dr = (dmax - dmin) * 0.1
    y0 = dmin
    y1 = (n_rows - 1) * dr + dmax
    ax.set_ylim(y0, y1)

    seqs = []
    for i in range(n_rows):
        seqs.append(np.column_stack((t, data[:, i])))
        ticktocks.append(i * dr)

    offsets = np.zeros((n_rows, 2), dtype=float)
    offsets[:, 1] = ticktocks

    lines = LineCollection(seqs, offsets=offsets, transOffset=None, linewidths=1, colors="black")
    ax.add_collection(lines)

    # Set the yticks to use axes coordinates on the y axis
    ax.tick_params(labelsize=40)
    ax.set_yticks(ticktocks)
    ax.set_yticklabels(columns, fontsize=35.0, fontweight='medium')
    ax.set_xlabel('Time (s)', fontsize=35.0, fontweight='medium')

    plt.tight_layout()

    # Show the major grid lines every tick
    plt.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.8, axis="x")

    # Show the minor grid lines with very faint and almost transparent grey lines
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2, axis="x")
    fig.suptitle(title, fontsize=35)


    if save:
        # Check if save folder exist, if not create it
        if not exists(path + task.capitalize() + "Task" + sep + new_folder):
            os.makedirs(path + task.capitalize() + "Task" + sep + new_folder)

        save_dir = path + task.capitalize() + "Task" + sep + new_folder + sep
        print(save_dir)
        # save pdf and svg
        plt.savefig(save_dir + name.split(".")[0] + ".pdf")
        plt.savefig(save_dir + name.split(".")[0] + ".svg")
    else:
        plt.show()


if __name__ == '__main__':
    # Complete Task
    # p = ".." + sep + ".." + sep + "Participants" + sep
    # One minute sequence of the task
    p = ".." + sep + ".." + sep + "Participants" + sep + "OneMin" + sep
    t = "pacman"
    for n, d in get_file(p, t):
        gen_plt(d, n, p, t, "PicturesBA", n.split(".")[0], save=True)
