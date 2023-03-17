import numpy as np
import pandas as pd
# import polars as pl
import utils.utilities as utils
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

# TODO change dfg to trace based instead of event
# 1. number of traces or number of timestamps -> get equivalent timespaces
# 2. get all traces that belong to one event based on timestamp

# TODO E2E preprocessing pipeline
# 1. get dfg matrix
# 2. calc similarity values with measure
# 3. save pictures in data folder according to tf.dataset structure

# TODO flip y-axis


def preprocessing_pipeline(n_windows=100, p_mode="train", window_mode="count"):

    # get all paths and file names of event logs
    log_files = utils.get_event_log_paths()

    # incrementally store number of log based on drift type - for file naming purposes
    log_numbers = defaultdict(lambda: 0)

    synthetic_log = True

    if p_mode == "eval":
        synthetic_log = False

    # iterate through log files
    for name, path in tqdm(log_files.items(), desc="Preprocessing Event Logs"
                           , unit="Event Log"):

        ######
        # TODO outsource to function
        # event_log_name = "event_" + name.split(".")[0]
        # log_info = logs_info.filter(pl.col("Event Log") == event_log_name)
        # drift_type = log_info["Drift Type"][0]

        # log_filter = logs_info.filter(pl.col("Drift Type") == drift_type)
        # log_filter = log_filter.insert_at_idx(0, pl.Series("idx", [i for i in range(
        #     1, len(log_filter) + 1)])).filter(pl.col("Event Log") == event_log_name)
        # log_number = log_filter["idx"][0]
        ######

        event_log = utils.import_event_log(path=path, name=name)

        # TODO outsource to function - save info as csv/dataframe
        if p_mode == "train":
            noise_info, drift_info = utils.get_nested_log_information(event_log)
            # log_number = drift_info["log_id"]
            drift_type = drift_info["drift_type"]
        elif p_mode == "eval":
            drift_type = "eval"    
        

        log_numbers[drift_type] += 1

        # if the log contains incomplete traces, the log is filtered
        filtered_log = utils.filter_complete_events(event_log)

        if window_mode == "time":
            # TODO implement log_to_windowed_dfg_timestamp
            pass
        elif window_mode == "count":
            windowed_dfg_matrices = log_to_windowed_dfg_count(
                filtered_log, n_windows, synthetic_log)

        sim_matrix = similarity_calculation(windowed_dfg_matrices)

        utils.matrix_to_img(matrix=sim_matrix, number=log_numbers[drift_type],
                            drift_type=drift_type, mode="color")


def log_to_windowed_dfg_timestamp():
    pass


def log_to_windowed_dfg_count(event_log, n_windows, synthetic_log):

    event_log_df = log_converter.apply(
        event_log, variant=log_converter.Variants.TO_DATA_FRAME)
    event_log_df = dataframe_utils.convert_timestamp_columns_in_df(
        event_log_df)
    
    # event_log_df["case:concept:name"] = event_log_df["case:concept:name"].apply(
    #     lambda x: ''.join(list(filter(str.isdigit, x))))

    act_names = np.unique(event_log_df["concept:name"])
    
    unique_traces = pd.unique(event_log_df["case:concept:name"])
    
    
    # 
    # unique_traces = [''.join(list(filter(str.isdigit, i))) for i in unique_traces]

    window_size = len(event_log) // n_windows

    freq_count = 0
    dfg_graphs = []

    left_boundary = 0
    right_boundary = window_size

    # start_act = []
    # end_act = []
    for i in range(1, n_windows + 1):
        dfg_matrix_df = pd.DataFrame(0, columns=act_names, index=act_names)

        

        if synthetic_log:
            if i < n_windows:
                log_window = event_log_df.loc[(event_log_df["case:concept:name"].astype(
                    int) >= left_boundary) & (event_log_df["case:concept:name"]
                                              .astype(int)
                                              < right_boundary)]
            else:
                log_window = event_log_df.loc[(event_log_df["case:concept:name"].astype(
                    int) >= left_boundary)]

        elif not synthetic_log:
            
            if i < n_windows:
            
                w_unique_traces = unique_traces[left_boundary:right_boundary]
                
            else:
                w_unique_traces = unique_traces[left_boundary:]
            
            log_window = event_log_df[event_log_df["case:concept:name"]
                                      .isin(w_unique_traces)]
            
            # log_window = event_log_df.query('case:concept:name in @w_unique_traces')
            
            # if i < n_windows:
            #     log_window = event_log_df.loc[(event_log_df["case:concept:name"]
            # .astype(
            #         int) >= min(w_unique_traces)) & (event_log_df["case:concept:name"]
            #                                          .astype(int)
            #                                          <= max(w_unique_traces))]
            # else:
            #     log_window = event_log_df.loc[(event_log_df["case:concept:name"]
            # .astype(
            #         int) >= min(w_unique_traces))]

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

    total_freq = check_dfg_graph_freq(event_log_df)

    assert total_freq == freq_count, "Missing directly follow relations"

    # print(total_freq == freq_count)
    # print("Total freq", total_freq)
    # print("Freq count", freq_count)

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
