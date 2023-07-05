import os
import json
import numpy as np
import pandas as pd
import cnn_image_detection.utils.utilities as cnn_utils
import object_segmentation.utils.utilities as seg_utils
import object_segmentation.utils.config as cfg
import object_segmentation.utils.vdd_helper as vdd_helper
import object_segmentation.utils.vdd_data_analysis as vdd

from pm4py import discover_dfg_typed
from numpy import linalg as LA
from scipy import spatial
from scipy.stats import wasserstein_distance
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.util import dataframe_utils
from cnn_image_detection.utils.sanity_checks import check_dfg_graph_freq
from tqdm import tqdm


def preprocessing_pipeline_multilabel(n_windows=100, p_mode="train"):
    # create experiment folder structure
    cfg.DEFAULT_DATA_DIR = cnn_utils.create_multilabel_experiment(cfg.DEFAULT_DATA_DIR)

    # get all paths and file names of event logs
    log_files = cnn_utils.get_event_log_paths(cfg.DEFAULT_LOG_DIR)

    drift_info = seg_utils.extract_drift_information(cfg.DEFAULT_LOG_DIR)

    # incrementally store number of log based on drift type - for file naming purposes
    drift_number = 1

    log_matching = {}
    window_info = {}
    date_info = {}

    # iterate through log files
    for name, path in tqdm(log_files.items(), desc="Preprocessing Event Logs",
                           unit="Event Log"):

        # load event log
        event_log = cnn_utils.import_event_log(path=path, name=name)

        # if the log contains incomplete traces, the log is filtered
        filtered_log = cnn_utils.filter_complete_events(event_log)

        windowed_dfg_matrices, borders, window_information, log_date_info = \
            log_to_windowed_dfg_count(filtered_log, n_windows)
        
        window_info[name] = window_information

        drift_info = seg_utils.update_trace_indices(drift_info, name, borders)

        # get similarity matrix
        sim_matrix = similarity_calculation(windowed_dfg_matrices)

        log_matching[name] = drift_number
        date_info[name] = log_date_info

        # save matrix as image
        seg_utils.matrix_to_img(matrix=sim_matrix, 
                                number=drift_number,
                                exp_path=cfg.DEFAULT_DATA_DIR,
                                mode=cfg.COLOR)

        # increment log number
        drift_number += 1

    drift_info.to_csv(os.path.join(cfg.DEFAULT_DATA_DIR, "drift_info.csv"))
    
    log_matching_df = pd.DataFrame.from_dict(log_matching, 
                                                orient="index", 
                                                columns=["image_id"])
    log_matching_df.to_csv(os.path.join(cfg.DEFAULT_DATA_DIR, "log_matching.csv"))
    
    window_info_path = os.path.join(cfg.DEFAULT_DATA_DIR, "window_info.json")
    with open(window_info_path, "w", encoding='utf-8') as file:
        json.dump(window_info, file)
        
    date_info_path = os.path.join(cfg.DEFAULT_DATA_DIR, "date_info.json")
    with open(date_info_path, "w", encoding='utf-8') as file:
        json.dump(date_info, file)

    seg_utils.generate_annotations(drift_info, 
                                   dir=cfg.DEFAULT_DATA_DIR,
                                   log_matching=log_matching,
                                   log_names=log_files.keys())

    if cfg.AUTOMATE_TFR_SCRIPT:
        seg_utils.start_tfr_script(repo_dir=cfg.TENSORFLOW_MODELS_DIR,
                                data_dir=cfg.DEFAULT_DATA_DIR,
                                tfr_dir=cfg.TFR_RECORDS_DIR,
                                prefix=cfg.OUTPUT_PREFIX)
        

def vdd_pipeline():
    # create experiment folder structure
    cfg.DEFAULT_DATA_DIR = vdd_helper.create_experiment(cfg.DEFAULT_DATA_DIR)

    # get all paths and file names of event logs
    log_files = cnn_utils.get_event_log_paths(cfg.DEFAULT_LOG_DIR)

    drift_info = vdd_helper.extract_vdd_drift_information(cfg.DEFAULT_LOG_DIR)

    bbox_df = pd.DataFrame()
    
    # incrementally store number of log based on drift type - for file naming purposes
    drift_number = 1

    log_matching = {}
    date_info = {}
    first_timestamps = {} 

    # iterate through log files
    for name, path in tqdm(log_files.items(), desc="Preprocessing Event Logs",
                           unit="Event Log"):
        
        log_path = os.path.join(path, name)
        
        # load event log
        event_log = cnn_utils.import_event_log(path=path, name=name)

        # if the log contains incomplete traces, the log is filtered
        filtered_log = cnn_utils.filter_complete_events(event_log)

        if cfg.MINE_CONSTRAINTS:
            minerful_csv_path = vdd_helper.vdd_mine_minerful_for_declare_constraints(
                name,
                log_path,
                cfg.DEFAULT_DATA_DIR
                )
        else:
            minerful_csv_path = vdd_helper.get_minerful_constraints_path(log_name=name,
                                                                         constraints_dir=cfg.CONSTRAINTS_DIR)

        ts_ticks = vdd_helper.vdd_save_separately_timestamp_for_each_constraint_window(
                filtered_log)
        
        first_timestamps[name] = vdd_helper.get_first_timestamp_per_trace(filtered_log)

        constraints = vdd_helper.vdd_import_minerful_constraints_timeseries_data(
            minerful_csv_path)

        # workaround
        try:
            constraints, \
                cluster_bounds, \
                horisontal_separation_bounds_by_cluster, \
                clusters_with_declare_names, \
                clusters_dict, \
                cluster_order = \
                vdd.do_cluster_changePoint(constraints, cp_all=cfg.CP_ALL)
        except ValueError:
            continue
            
        timestamps = vdd_helper.get_drift_moments_timestamps(log_name=name, 
                                                             drift_info=drift_info)
        
        drift_types = vdd_helper.get_drift_types(log_name=name,
                                                 drift_info=drift_info)
        
        bboxes, fig_bbox, log_date_info = vdd_helper.vdd_draw_drift_map_with_clusters(
            data=constraints,
            number=drift_number,
            exp_path=cfg.DEFAULT_DATA_DIR,
            ts_ticks=ts_ticks,
            timestamps=timestamps,
            drift_types=drift_types)
        
        bbox_df = vdd_helper.update_bboxes_for_vdd(bbox_df, bboxes, name, fig_bbox)
            
        log_matching[name] = drift_number
        date_info[name] = log_date_info
        
        # increment log number
        drift_number += 1
        
    drift_info = vdd_helper.merge_bboxes_with_drift_info(bbox_df, drift_info)
    
    drift_info.to_csv(os.path.join(cfg.DEFAULT_DATA_DIR, "drift_info.csv"))
    
    log_matching_df = pd.DataFrame.from_dict(log_matching, 
                                                orient="index", 
                                                columns=["image_id"])
    log_matching_df.to_csv(os.path.join(cfg.DEFAULT_DATA_DIR, "log_matching.csv"))
    
    date_info_path = os.path.join(cfg.DEFAULT_DATA_DIR, "date_info.json")
    with open(date_info_path, "w", encoding='utf-8') as file:
        json.dump(date_info, file)
    
    first_timestamps_path = os.path.join(cfg.DEFAULT_DATA_DIR, "first_timestamps.json")
    with open(first_timestamps_path, "w", encoding='utf-8') as file:
        json.dump(first_timestamps, file)
    
    vdd_helper.generate_vdd_annotations(drift_info, 
                                   dir=cfg.DEFAULT_DATA_DIR,
                                   log_matching=log_matching,
                                   log_names=log_matching.keys())

    if cfg.AUTOMATE_TFR_SCRIPT:
        seg_utils.start_tfr_script(repo_dir=cfg.TENSORFLOW_MODELS_DIR,
                                data_dir=cfg.DEFAULT_DATA_DIR,
                                tfr_dir=cfg.TFR_RECORDS_DIR,
                                prefix=cfg.OUTPUT_PREFIX)


def log_to_windowed_dfg_count(event_log, n_windows):

    # convert event log to pandas dataframe
    event_log_df = log_converter.apply(
        event_log, variant=log_converter.Variants.TO_DATA_FRAME)
    event_log_df = dataframe_utils.convert_timestamp_columns_in_df(
        event_log_df, timest_format="ISO8601")
    
    min_date = np.min(event_log_df["time:timestamp"])
    max_date = np.max(event_log_df["time:timestamp"])
    date_info = (seg_utils.datetime_2_str(min_date), 
                 seg_utils.datetime_2_str(max_date))
    
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
    borders = []
    dfg_graphs = []
    # start_act = []
    # end_act = []

    window_information = {}

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

        borders.append((left_boundary, right_boundary))

        left_boundary = right_boundary
        right_boundary += window_size

        # get id of first trace in window - for evaluation
        first_trace = w_unique_traces[0]
        first_timestamp = min(
            event_log_df.loc[event_log_df["case:concept:name"] == first_trace,
                             "time:timestamp"])
        first_timestamp = first_timestamp.strftime('%Y-%m-%d %H:%M:%S')
        window_information[i] = (first_trace, first_timestamp)

    # compare dfg frequencies of all windows with dfg frequencies of complete log
    # ensures that there are no missing relations
    total_freq = check_dfg_graph_freq(event_log_df)
    assert total_freq == freq_count, (
        "Missing directly follow relations.\n"
        f"Number of relations in whole event log: {total_freq}.\n"
        f"Number of relations in all windows: {freq_count}"
    )
    
    # , np.array(start_act), np.array(end_act)
    return np.array(dfg_graphs), borders, window_information, date_info


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
                sim_matrix[i, j] = calc_distance_norm(matrix_i, 
                                                      matrix_j, 
                                                      cfg.DISTANCE_MEASURE)
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


def calc_distance_norm(matrix_1, matrix_2, option):
    diff = matrix_1 - matrix_2
    if option == "fro":
        # Frobenius norm
        dist_value = LA.norm(diff, "fro")
    elif option == "nuc":
        # nuclear norm
        dist_value = LA.norm(diff, "nuc")
    elif option == "inf":
        # max norm
        dist_value = LA.norm(diff, np.inf)
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


if __name__ == "__main__":
    
    preprocessing_pipeline_multilabel(cfg.N_WINDOWS)