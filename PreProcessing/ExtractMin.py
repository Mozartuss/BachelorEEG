import os
from os.path import sep, exists


def extract_minute(folder, file_name, data):
    task = file_name.split("_")[-1].split(".")[0].capitalize()
    participant = file_name.split("_")[0]

    if task == "Pacman":
        start_time = data.iloc[0][0].astype(int) + 3 * 60
        end_time = start_time + 60
    elif task == "Video":
        start_time = data.iloc[0][0].astype(int) + (3 * 60 + 30)
        end_time = start_time + 60
    else:
        print("Task isn't correct")
        exit()

    index_start = data.loc[data.iloc[:, 0].astype(int) == start_time].index[0]
    index_end = data.loc[data.iloc[:, 0].astype(int) == end_time].index[0]

    if abs((index_end - index_start) - 15000) != 0:
        index_end += 15000 - (index_end - index_start)

    one_minute_df = data.loc[index_start:index_end]
    one_minute_df_cut = one_minute_df.iloc[:, :17]

    if not exists(folder + "OneMin" + sep + task + "Task"):
        os.makedirs(folder + "OneMin" + sep + task + "Task")

    save_path = folder + "OneMin" + sep + task + "Task" + sep + file_name
    one_minute_df_cut.to_csv(save_path, index=False,
                             header=["Timestamp", "FP1", "FP2", "F3", "FZ", "F4", "T7", "C3", "CZ", "C4", "T8", "P3",
                                     "PZ", "P4", "PO7", "PO8", "OZ"])
    print("Save:", save_path)


if __name__ == '__main__':
    f = ".." + sep + ".." + sep + "Participants" + sep
    # for f_n, d in get_file(f, option="pacman"):
    #    extract_minute(f, f_n, d)
