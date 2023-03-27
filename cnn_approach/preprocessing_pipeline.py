import numpy as np
import pandas as pd
# import polars as pl
import utils.utilities as utils
import utils.config as cfg
# import pm4py
from pm4py import discover_dfg_typed
from numpy import linalg as LA
from scipy import spatial
from scipy.stats import wasserstein_distance
# from pm4py.algo.filtering.log.attributes import attributes_filter
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.util import dataframe_utils
from utils.sanity_checks import check_dfg_graph_freq
from collections import defaultdict
from tqdm import tqdm
# from scipy import sparse

# TODO change dfg to trace based instead of event
# 1. number of traces or number of timestamps -> get equivalent timespaces
# 2. get all traces that belong to one event based on timestamp

# TODO E2E preprocessing pipeline
# 1. get dfg matrix
# 2. calc similarity values with measure
# 3. save pictures in data folder according to tf.dataset structure

# TODO flip y-axis


def preprocessing_pipeline(n_windows=100, p_mode="train"):
    # create experiment folder structure
    cfg.DEFAULT_DATA_DIR = utils.create_experiment()

    # get all paths and file names of event logs
    log_files = utils.get_event_log_paths()

    # incrementally store number of log based on drift type - for file naming purposes
    log_numbers = defaultdict(lambda: 0)

    # iterate through log files
    for name, path in tqdm(log_files.items(), desc="Preprocessing Event Logs",
                           unit="Event Log"):

        # load event log
        event_log = utils.import_event_log(path=path, name=name)

        # TODO outsource to function - save info as csv/dataframe
        if p_mode == "train":
            noise_info, drift_info = utils.get_nested_log_information(
                event_log)
            # log_number = drift_info["log_id"]
            drift_type = drift_info["drift_type"]
        elif p_mode == "eval":
            drift_type = "eval"

        # increment log number
        log_numbers[drift_type] += 1

        # if the log contains incomplete traces, the log is filtered
        filtered_log = utils.filter_complete_events(event_log)

        windowed_dfg_matrices = log_to_windowed_dfg_count(
            filtered_log, n_windows)

        # get similarity matrix
        sim_matrix = similarity_calculation(windowed_dfg_matrices)

        # save matrix as image
        utils.matrix_to_img(matrix=sim_matrix, number=log_numbers[drift_type],
                            drift_type=drift_type, exp_path=cfg.DEFAULT_DATA_DIR,
                            mode="color")


def log_to_windowed_dfg_count(event_log, n_windows):

    # convert event log to pandas dataframe
    event_log_df = log_converter.apply(
        event_log, variant=log_converter.Variants.TO_DATA_FRAME)
    event_log_df = dataframe_utils.convert_timestamp_columns_in_df(
        event_log_df)

    # get unique event names
    act_names = np.unique(event_log_df["concept:name"])
    # idx_names = np.arange(len(act_names))

    # get unique trace names
    # hint: pandas unique does not sort the result, therefore it is faster and the
    # chronological order is maintained
    unique_traces = pd.unique(event_log_df["case:concept:name"])

    # get window size based on number of windows and event log size
    # event log size is equal to number of traces
    window_size = len(event_log) // n_windows

    # initialize helper variables
    freq_count = 0
    left_boundary = 0
    right_boundary = window_size
    dfg_graphs = []
    # start_act = []
    # end_act = []

    # iterate through windows
    for i in range(1, n_windows + 1):
        dfg_matrix_df = pd.DataFrame(0, columns=act_names, index=act_names)

        # get all trace names that are in selected window, traces are sorted by
        # timestamp
        if i < n_windows:

            w_unique_traces = unique_traces[left_boundary:right_boundary]

        # at last window fill until the end of the list
        else:
            w_unique_traces = unique_traces[left_boundary:]

        # search all events for given traces
        log_window = event_log_df[event_log_df["case:concept:name"]
                                  .isin(w_unique_traces)]

        # get dfg graph for window
        graph, sa, ea = discover_dfg_typed(log_window)

        # transform dfg graph into dfg matrix
        for relation, freq in graph.items():
            rel_a, rel_b = relation
            dfg_matrix_df.at[rel_a, rel_b] = freq

            freq_count += freq

        dfg_graphs.append(dfg_matrix_df)
        # start_act.append(sa)
        # end_act.append(ea)

        left_boundary = right_boundary
        right_boundary += window_size

    # compare dfg frequencies of all windows with dfg frequencies of complete log
    # ensures that there are no missing relations
    total_freq = check_dfg_graph_freq(event_log_df)
    assert total_freq == freq_count, (
        "Missing directly follow relations.\n"
        f"Number of relations in whole event log: {total_freq}.\n"
        f"Number of relations in all windows: {freq_count}"
    )

    return np.array(dfg_graphs)  # , np.array(start_act), np.array(end_act)


def similarity_calculation(windowed_dfg):

    # create matrix
    n = len(windowed_dfg)
    sim_matrix = np.zeros((n, n))

    # calculate similarity measure between all elements of dfg matrix
    for i, matrix_i in enumerate(windowed_dfg):
        for j, matrix_j in enumerate(windowed_dfg):
            if (i == j) or (sim_matrix[i, j] != 0):
                continue
            else:
                sim_matrix[i, j] = calc_distance_norm(matrix_i, matrix_j)
                sim_matrix[j, i] = sim_matrix[i, j]
            # sim_matrix[i, j] = calc_distance_norm(matrix_i, matrix_j)

    # check diagonal of similarity matrix
    assert np.sum(np.diagonal(sim_matrix)
                  ) == 0, "The diagonal of the similarity matrix must be zero."

    # normalize similarity matrix
    norm_sim_matrix = 1 - sim_matrix / np.amax(sim_matrix)

    # transform matrix values to color integers
    img_matrix = np.uint8(norm_sim_matrix * 255)

    # stacked_img = np.stack((img_matrix,)*3, axis=-1)
    # data = Image.fromarray(img_matrix)
    # data.show()
    # from matplotlib import pyplot as plt
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
