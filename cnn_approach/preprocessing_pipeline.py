import numpy as np
import pandas as pd
import polars as pl
import utils.utilities as utils
from pm4py import discover_dfg_typed
from numpy import linalg as LA
from scipy import spatial
from scipy.stats import wasserstein_distance


# TODO change dfg to trace based instead of event
# 1. number of traces or number of timestamps -> get equivalent timespaces
# 2. get all traces that belong to one event based on timestamp

# TODO compute p-values (?)

# TODO E2E preprocessing pipeline
# 1. get dfg matrix
# 2. calc similarity values with measure
# 3. save pictures in data folder according to tf.dataset structure

# TODO preprocess info table from cdlg

# TODO flip y-axis


def preprocessing_pipeline(n_windows: int):
    # event_log = pm.read_xes(event_log_path)

    log_files = utils.get_event_log_paths()
    logs_info = utils.get_collection_information()

    for name, path in log_files.items():

        ######
        # TODO outsource to function
        event_log_name = "event_" + name.split(".")[0]
        log_info = logs_info.filter(pl.col("Event Log") == event_log_name)
        drift_type = log_info["Drift Type"][0]

        log_filter = logs_info.filter(pl.col("Drift Type") == drift_type)
        log_filter = log_filter.insert_at_idx(0, pl.Series("idx", [i for i in range(
            1, len(log_filter) + 1)])).filter(pl.col("Event Log") == event_log_name)
        log_number = log_filter["idx"][0]
        ######

        event_log = utils.read_event_log(path=path, name=name)

        windowed_dfg_matrices = log_to_windowed_dfg_count(event_log, n_windows)

        sim_matrix = similarity_calculation(windowed_dfg_matrices)

        utils.matrix_to_img(sim_matrix, log_number, drift_type, mode="color")

    # return sim_matrices


def log_to_windowed_dfg_timestamp():
    pass


def log_to_windowed_dfg_count(event_log, n_windows):

    act_names = np.unique(event_log["concept:name"])

    # print("Loglenth: ", len(event_log))

    freq_count = 0

    window_size = len(event_log) // n_windows + 1
    dfg_graphs = []
    # start_act = []
    # end_act = []
    left_boundary = 0
    right_boundary = window_size
    for _ in range(1, n_windows + 1):
        dfg_matrix_df = pd.DataFrame(0, columns=act_names, index=act_names)

        log_window = event_log[left_boundary:right_boundary]

        graph, sa, ea = discover_dfg_typed(log_window)

        for relation, freq in graph.items():
            rel_a, rel_b = relation
            dfg_matrix_df.at[rel_a, rel_b] = freq

            freq_count += freq

        dfg_graphs.append(dfg_matrix_df)
        # start_act.append(sa)
        # end_act.append(ea)

        left_boundary = right_boundary
        right_boundary += window_size

    # print("Freq_count: ", freq_count)

    return np.array(dfg_graphs)  # , np.array(start_act), np.array(end_act)


def similarity_calculation(windowed_dfg):

    # create matrix
    n = len(windowed_dfg)
    sim_matrix = np.zeros((n, n))

    for i, matrix_i in enumerate(windowed_dfg):
        for j, matrix_j in enumerate(windowed_dfg):
            sim_matrix[i, j] = calc_distance_norm(matrix_i, matrix_j)

    # check diagonal of similarity matrix
    assert np.sum(np.diagonal(sim_matrix)
                  ) == 0, "The diagonal of the similarity matrix must be zero."

    norm_sim_matrix = 1 - sim_matrix / np.amax(sim_matrix)

    img_matrix = np.uint8(norm_sim_matrix * 255)
    # print(img_matrix.shape)

    # stacked_img = np.stack((img_matrix,)*3, axis=-1)

    # print(stacked_img.shape)

    # data = Image.fromarray(img_matrix)
    # data.show()
    # plt.imshow(img_matrix, interpolation='nearest')
    # plt.show()

    return img_matrix


def calc_distance_norm(matrix_1, matrix_2, option="fro"):
    diff = matrix_1 - matrix_2
    if option == "fro":
        # Frobenius norm
        dist_value = LA.norm(diff, "fro")
    elif option == "nuc":
        # nuclear norm
        dist_value = LA.norm(diff, "nuc")
    elif option == "inf":
        # max norm
        dist_value = LA.norm(diff, "inf")
    elif option == "l2":
        # L2 norm
        dist_value = LA.norm(diff, 2)
    elif option == "cos":
        # cosine distance
        dist_value = spatial.distance.cosine(
            matrix_1.ravel(), matrix_2.ravel())
    elif option == "earth":
        # earth mover distance == wasserstein distance
        dist_value = 0

        # compute histograms and wasserstein distance for each activity
        for i in range(0, len(matrix_1)):
            matrix_1_hist, _ = np.histogram(
                matrix_1[i, :], bins=np.arange(-0.5, len(matrix_1)), density=True)
            matrix_2_hist, _ = np.histogram(
                matrix_2[i, :], bins=np.arange(-0.5, len(matrix_2)), density=True)
            dist_value += wasserstein_distance(matrix_1_hist, matrix_2_hist)

    if np.isnan(dist_value):
        dist_value = LA.norm(diff, "fro")

    return dist_value
