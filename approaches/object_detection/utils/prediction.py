import os
import json
import tensorflow as tf
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

import utils.preprocessing as pp
import utils.evaluation as eval
import utils.utilities as utils
import utils.vdd_helper as vdd_helper
import utils.vdd_data_analysis as vdd

from tqdm import tqdm
from typing import List


def prediction_pipeline(log_dir: str, encoding_type: str, output_path: str,
                        cp_all=None, n_windows=None) -> str:
    """Entry script for prediction preprocessing pipeline.

    Args:
        log_dir (str): Event log directory
        encoding_type (str): Preprocessing encoding method
        output_path (str): Output directory
        cp_all (bool, optional): VDD measure. Defaults to None.
        n_windows (int, optional): Number of windows for WINSIM. Defaults to None.

    Raises:
        ValueError: Raises ValueError if no valid encoding type is specified

    Returns:
        str: Image directory
    """
    image_dir = os.path.join(output_path, f"{encoding_type}_images")

    if not os.path.isdir(image_dir):
        os.makedirs(image_dir)

    if encoding_type == "vdd":
        vdd_pipeline(log_dir, output_path=image_dir, cp_all=cp_all)
    elif encoding_type == "winsim":
        winsim_pipeline(log_dir, output_path=image_dir,
                        n_windows=n_windows)
    else:
        raise ValueError("Please specify a valid encoding type. \n \
                         Valid types: ('vdd' or 'winsim')")
    return image_dir


def vdd_pipeline(log_dir: str, output_path: str,
                 cp_all: bool):
    """VDD pipeline for encoding logs for prediction. 
    When using this function please make sure to set the path of your MINERful 
    distribution in the MINERFUL_SCRIPTS_DIR variable and WINDOW_SYSTEM 
    (to indicate whether you are using windows or not) in the object_segmentation 
    config file. Also feel free to adapt the corresponding VDD values SUB_L and SLI_BY.

    Args:
        log_dir (str): Log directory
        output_path (str): Output path
        cp_all (bool): VDD measure
    """
    # get all paths and file names of event logs
    log_files = utils.get_event_log_paths(log_dir)

    date_info = {}
    first_timestamps = {}

    # iterate through log files
    for name, path in tqdm(log_files.items(), desc="Preprocessing Event Logs",
                           unit="Event Log"):

        log_path = os.path.join(path, name)
        real_name = name.split(".")[0]

        # load event log
        event_log = utils.import_event_log(path=path, name=name)

        # if the log contains incomplete traces, the log is filtered
        filtered_log = utils.filter_complete_events(event_log)

        minerful_csv_path = vdd_helper.vdd_mine_minerful_for_declare_constraints(
            name,
            log_path,
            output_path
        )

        ts_ticks = vdd_helper.vdd_save_separately_timestamp_for_each_constraint_window(
            filtered_log)

        first_timestamps[real_name] = vdd_helper.get_first_timestamp_per_trace(
            filtered_log)

        constraints = vdd_helper.vdd_import_minerful_constraints_timeseries_data(
            minerful_csv_path)

        try:
            constraints, \
                _, \
                _, \
                _, \
                _, \
                _ = \
                vdd.do_cluster_changePoint(constraints, cp_all=cp_all)
        # In some edge cases the change points can not be determined
        # The error occurs only extremely rarely and is therefore skipped
        except ValueError:
            continue

        log_date_info = vdd_helper.vdd_draw_drift_map_prediction(
            data=constraints,
            number=real_name,
            exp_path=output_path,
            ts_ticks=ts_ticks)

        date_info[real_name] = log_date_info

    date_info_path = os.path.join(output_path, "date_info.json")
    with open(date_info_path, "w", encoding='utf-8') as file:
        json.dump(date_info, file)

    first_timestamps_path = os.path.join(
        output_path, "first_timestamps.json")
    with open(first_timestamps_path, "w", encoding='utf-8') as file:
        json.dump(first_timestamps, file)


def winsim_pipeline(log_dir: str, output_path: str, n_windows: int):
    """WINSIM pipeline for encoding logs for prediction. 

    Args:
        log_dir (str): Log directory
        output_path (str): Output path
        n_windows (int): Number of windows to split log
    """
    # get all paths and file names of event logs
    log_files = utils.get_event_log_paths(log_dir)

    window_info = {}

    # iterate through log files
    for name, path in tqdm(log_files.items(), desc="Preprocessing Event Logs",
                           unit="Event Log"):
        real_name = name.split(".")[0]
        # load event log
        event_log = utils.import_event_log(path=path, name=name)

        # if the log contains incomplete traces, the log is filtered
        filtered_log = utils.filter_complete_events(event_log)

        windowed_dfg_matrices, _, window_information, _ = \
            pp.log_to_windowed_dfg_count(filtered_log, n_windows)

        window_info[real_name] = window_information

        # get similarity matrix
        sim_matrix = pp.similarity_calculation(windowed_dfg_matrices)

        # save matrix as image
        utils.matrix_to_img(matrix=sim_matrix,
                            number=real_name,
                            exp_path=output_path,
                            mode="color")

    window_info_path = os.path.join(output_path, "window_info.json")
    with open(window_info_path, "w", encoding='utf-8') as file:
        json.dump(window_info, file)


def save_pred_results(results: dict, output_path: str):
    """Saves prediction results to output path.

    Args:
        results (dict): Evaluation measures
        output_path (str): Evaluation directory
    
    Returns:
        pd.DataFrame: DataFrame, containing predictions
    """
    results_df = pd.DataFrame.from_dict(results, orient="index")
    save_path = os.path.join(output_path, "prediction_results.csv")
    results_df.to_csv(save_path, sep=",")


def get_changepoints_trace_idx_winsim(bboxes: list, y_pred: list,
                                      window_info: dict) -> List[tuple]:
    """Get changepoints as trace indices for WINSIM encoding. 

    Args:
        bboxes (list): List of bboxes
        y_pred (list): List of predictions
        window_info (dict): Window info

    Returns:
        List[tuple]: List of changepoints as trace indices
    """
    change_points = []
    if len(bboxes) == 0:
        return change_points
    else:
        for i, bbox in enumerate(bboxes):
            if y_pred[i] == "sudden":
                # changepoint is equal to the date of the first trace in the middle
                # window of bbox
                change_point = eval.get_sudden_changepoint_winsim(
                    round(bbox[0]))
                change_point_trace_id = (window_info[str(change_point)][0],
                                         window_info[str(change_point)][0])
            else:
                # change start and end is equal to the date of the first trace in window
                # set window id of change start to at least 1 for edge cases
                # set window id of change end to maximum 200 for edge cases
                change_start = (1 if round(bbox[0]) < 1 else round(bbox[0]))
                change_end = (200 if round(bbox[2]) > 200 else round(bbox[2]))
                change_point_trace_id = (window_info[str(change_start)][0],
                                         window_info[str(change_end)][0])
            change_points.append(change_point_trace_id)
        return change_points


def get_changepoints_trace_idx_vdd(bboxes: list, y_pred: list,
                                   timestamps_per_trace: dict,
                                   min_date: dt.date, max_date: dt.date,
                                   targetsize: int) -> List[tuple]:
    """Get changepoints as trace indices for VDD encoding. 
    Changepoint is equal to the shift of the relative position starting from the 
    min-date. The time delta is calculated from the relative position, 
    since the x-axis represents the time period. The trace index searched for is the 
    trace index whose timestamp is closest to the calculated timestamp.

    Args:
        bboxes (list): List of bboxes
        y_pred (list): List of predictions
        timestamps_per_trace (dict): First timestamp for each trace of event log
        min_date (dt.date): Minimum date in event log
        max_date (dt.date): Maximum date in event log
        targetsize (int): Image targetsize

    Returns:
        List[tuple]: List of changepoints as trace indices
    """
    change_points = []
    if len(bboxes) == 0:
        return change_points
    else:
        day_delta = max_date - min_date
        for i, bbox in enumerate(bboxes):
            if y_pred[i] == "sudden":
                xmin = eval.get_sudden_changepoint_vdd(int(bbox[0]))
                relative_xmin = xmin / targetsize
                change_point_date = min_date + dt.timedelta(days=int(day_delta.days *
                                                                     relative_xmin))
                closest_trace = get_closest_trace_index(change_point_date,
                                                        timestamps_per_trace)
                change_point_index = (closest_trace, closest_trace)
            else:
                xmin = bbox[0]
                xmax = bbox[2]
                relative_xmin = xmin / targetsize
                relative_xmax = xmax / targetsize
                change_start_date = min_date + dt.timedelta(days=int(day_delta.days *
                                                                     relative_xmin))
                change_end_date = min_date + dt.timedelta(days=int(day_delta.days *
                                                                   relative_xmax))
                change_start_index = get_closest_trace_index(change_start_date,
                                                             timestamps_per_trace)
                change_end_index = get_closest_trace_index(change_end_date,
                                                           timestamps_per_trace)
                change_point_index = (change_start_index, change_end_index)
            change_points.append(change_point_index)
        return change_points


def get_closest_trace_index(drift_moment_date: dt.date,
                            timetamps_per_trace: dict) -> int:
    """Determines the trace ID closest to timestamp.

    Args:
        drift_moment_date (dt.date): Predicted drift moment
        timetamps_per_trace (dict): First timestamp per trace of event log

    Returns:
        int: Trace ID
    """
    timestamps_df = pd.DataFrame.from_dict(timetamps_per_trace,
                                           orient="index",
                                           columns=["timestamp"])
    timestamps_df = timestamps_df.rename_axis("trace_id").reset_index()
    timestamps_df["timestamp"] = timestamps_df["timestamp"].apply(lambda _:
        dt.datetime.strptime(_, "%m-%d-%Y").date())

    index = timestamps_df.loc[timestamps_df["timestamp"] ==
                              eval.nearest(timestamps_df["timestamp"].to_list(),
                                           drift_moment_date)].index[0]

    return timestamps_df.iloc[index]["trace_id"]


def visualize_prediction(path: str, image: np.ndarray, image_name: str,
                         bbox_pred: np.ndarray, y_pred: np.ndarray, score: np.ndarray,
                         encoding: str):
    """Visualize predicted bounding boxes for an image.

    Args:
        path (str): Output path
        image (np.ndarray): Image as numpy array
        image_name (str): Image name
        bbox_pred (list): Predicted bounding box
        y_pred (list): Predicted classes
        score (list): Confidence score
        encoding (str): Encoding type
    """
    category_index, _ = utils.get_ex_decoder()

    plt.figure(figsize=(10, 10))
    
    image = image[0].numpy()

    utils.visualize_boxes_and_labels(image=image,
                                     bboxes=bbox_pred,
                                     labels=y_pred,
                                     score=score,
                                     category_index=category_index,
                                     is_groundtruth=False,
                                     encoding=encoding)
    plt.imshow(image)
    plt.axis('off')

    plt.savefig(os.path.join(path, f"{image_name}.png"),
                bbox_inches="tight")


def predict(image_dir: str, output_path: str, model: tf.keras.Model,
            encoding_type: str, n_windows=None):
    """Main prediction script.

    Args:
        image_dir (str): Directory containing images for prediction
        output_path (str): Output path
        model (tf.keras.Model): Trained model
        encoding_type (str): Name of encoding method
        n_windows (int, optional): Number of windows for WINSIM. Defaults to None.

    Raises:
        ValueError: Raises ValueError if no valid encoding type is specified
    """
    input_image_size = (256, 256)
    targetsize = 256
    threshold = 0.5
    model_fn = model.signatures['serving_default']
    pred_results = {}

    if encoding_type == "winsim":
        window_info = eval.get_window_info(image_dir)
    elif encoding_type == "vdd":
        timestamps_per_trace = eval.get_first_timestamps_vdd(image_dir)
        date_info = eval.get_date_info(image_dir)
    else:
        raise ValueError("Please specify a valid encoding type. \n \
                         Valid types: ('vdd' or 'winsim')")

    category_index, _ = utils.get_ex_decoder()

    images = eval.get_image_paths(image_dir)

    for image_name, image_path in tqdm(images.items(),
                                       desc="Detecting Concept Drift", unit="images"):

        path = os.path.join(image_path, image_name)
        image_name = image_name.split(".")[0]
        image = utils.load_image(path)
        image = utils.build_inputs_for_object_detection(
            image, input_image_size)
        image = tf.expand_dims(image, axis=0)
        image = tf.cast(image, dtype=tf.uint8)
        result = model_fn(image)

        scores = result['detection_scores'][0].numpy()
        confidence_scores = scores[scores > threshold]

        bbox_pred = result['detection_boxes'][0].numpy()
        bbox_pred = bbox_pred[scores > threshold]

        y_pred = result['detection_classes'][0].numpy().astype(int)
        y_pred = y_pred[scores > threshold]

        y_pred_category = eval.get_predicted_classes(y_pred, category_index)

        visualize_prediction(path=output_path,
                             image=image,
                             image_name=image_name,
                             bbox_pred=bbox_pred,
                             y_pred=y_pred,
                             score=confidence_scores,
                             encoding=encoding_type)

        if encoding_type == "winsim":
            bbox_pred = bbox_pred / targetsize \
                * n_windows
            log_window_info = window_info[image_name]
            pred_change_points = get_changepoints_trace_idx_winsim(
                bbox_pred, y_pred_category, log_window_info)
        elif encoding_type == "vdd":
            min_date, max_date = date_info[image_name]
            pred_change_points = get_changepoints_trace_idx_vdd(bboxes=bbox_pred,
                                                                y_pred=y_pred_category,
                                                                timestamps_per_trace=timestamps_per_trace[
                                                                    image_name],
                                                                min_date=eval.str_2_date(
                                                                    min_date),
                                                                max_date=eval.str_2_date(
                                                                    max_date),
                                                                targetsize=targetsize)
        pred_results[image_name] = \
            {"Detected Changepoints": pred_change_points,
                "Detected Drift Types": y_pred_category,
                "Prediction Confidence": np.round(confidence_scores, decimals=4)}

    save_pred_results(pred_results, output_path)
