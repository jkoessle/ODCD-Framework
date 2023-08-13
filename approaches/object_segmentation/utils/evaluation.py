import os
import re
import ast
import json
import shutil
import subprocess
import signal
import pandas as pd
import tensorflow as tf
import numpy as np
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt

from collections import defaultdict
from tqdm import tqdm
from typing import List, Tuple, Union
from math import inf

from . import config as cfg
from . import utilities as utils
from . import cdrift_evaluation as cdrift


def get_evaluation_metrics(y_true: list, y_pred: list,
                           y_true_label: list, y_pred_label: list,
                           factor: float, number_of_traces: int) \
        -> Tuple[dict, list, list]:
    """Get all relevant evaluation metrics.

    Args:
        y_true (list): List of groundtruth values
        y_pred (list): List of prediction values
        y_true_label (list): List of groundtruth labels
        y_pred_label (list): List of predicted labels
        factor (float): Factor for relative lag
        number_of_traces (int): Number of traces for event log

    Returns:
        Tuple[dict,list,list]: Dict, containing evaluation measures. 
        And sorted lists containing predicted and groundtruth labels
    """
    lag = int(factor * number_of_traces)

    _, assignments = cdrift.getTP_FP(detected=y_pred,
                                     known=y_true,
                                     lag=lag)

    matched_assignments = match_labels(assignments, y_true, y_true_label,
                                       y_pred, y_pred_label)

    if len(assignments) > 0:
        preds, trues = zip(*assignments)
    else:
        preds = trues = []

    sorted_trues = sorted(
        y_true, key=lambda x: trues.index(x) if x in trues else inf)

    y_pred_label_sorted = [y_pred_label[y_pred.index(x)] for x in preds]
    y_true_label_sorted = [y_true_label[y_true.index(x)] for x in sorted_trues]

    while len(y_pred_label_sorted) < len(y_true_label_sorted):
        y_pred_label_sorted.append(np.NaN)

    tp = len(matched_assignments)
    fp = len(y_pred) - tp

    f1, precision, recall = get_f1_score(tp, fp, len(y_true))
    average_lag = get_average_lag(matched_assignments)

    metrics = {"f1": f1,
               "precision": precision,
               "recall": recall,
               "lag": average_lag}
    return metrics, y_pred_label_sorted, y_true_label_sorted


def match_labels(assignments: list, y_true: list, y_true_labels: list,
                 y_pred: list, y_pred_labels: list) -> list:
    """Match predicted label with actual label. 
    If predicted label is not equal to actual label, 
    remove prediction from assignments.
    In other words: prediction becomes a false positive.

    Args:
        assignments (list): Assignments from CDrift evaluation
        y_true (list): Actual changepoints
        y_true_labels (list): Actual labels
        y_pred (list): Predicted changepoints
        y_pred_labels (list): Predicted labels

    Returns:
        list: Label-matched assignments
    """
    matched_assignments = []
    for assignment in assignments:
        pred, true = assignment
        if y_pred_labels[y_pred.index(pred)] == y_true_labels[y_true.index(true)]:
            matched_assignments.append(assignment)
    return matched_assignments


def get_precision(tp: int, fp: int) -> float:
    """Get precision.

    Args:
        tp (int): Number of true positives
        fp (int): Number of false positives

    Returns:
        float: Precision
    """
    return tp / (tp+fp)


def get_recall(tp: int, y_true_size: int) -> float:
    """Get recall.

    Args:
        tp (int): Number of true positives
        y_true_size (int): Size of true values

    Returns:
        float: Recall
    """
    return tp / y_true_size


def get_f1_score(tp: int, fp: int, y_true_size: int) -> Tuple[float, float, float]:
    """Get F1-score.

    Args:
        tp (int): Number of true positives
        fp (int): Number of false positives
        y_true_size (int): Size of true values

    Returns:
        Union[Tuple[float,float,float], Tuple[np.nan, np.nan, np.nan]]: F1-score
    """
    try:
        precision = get_precision(tp, fp)
        recall = get_recall(tp, y_true_size)

        f1_score = (2 * precision * recall) / (precision + recall)
        return f1_score, precision, recall
    except ZeroDivisionError:
        return np.nan, np.nan, np.nan


def get_average_lag(assignments: List[Tuple[int, int]]):
    """Source: 
    https://github.com/cpitsch/cdrift-evaluation/blob/main/cdrift/evaluation.py
    Author: cpitsch
    Note: The code was adapted so that it also works with two-dimensional tuples or 
    change points. Therefore, the description of the function has also been changed.
    Calculates the average lag between detected and actual start and end changepoints 
    (Caution: false positives do not affect this metric!)

    Args:
        assignments (List[Tuple[int,int]]): List of actual and detected changepoint 
            assignments

    Returns:
        float: the average distance between detected changepoints and the actual 
            changepoint they get assigned to
    """
    avg_lag = 0
    for (dc, ap) in assignments:
        avg_lag += abs(dc[0] - ap[0])
        avg_lag += abs(dc[1] - ap[1])
    try:
        return avg_lag / len(assignments)
    except ZeroDivisionError:
        return np.nan
    
    
def get_FP_TP_per_class(y_pred: list, y_true: list) -> dict:
    """Get true positives, false positives and false negatives per class.

    Args:
        y_pred (list): Predictions (classes)
        y_true (list): Groundtruth (classes)

    Returns:
        dict: Dictionary containing TP, FN and FP per class
    """
    measures_per_class = defaultdict(lambda: defaultdict(int))
    for i, true in enumerate(y_true):
        if true == y_pred[i]:
            measures_per_class[true]["TP"] += 1
        elif pd.isna(y_pred[i]):
            measures_per_class[true]["FN"] += 1
        else:
            measures_per_class[true]["FP"] += 1
    return dict(measures_per_class)


def get_f1_score_per_class(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """Get F1-score per class.

    Args:
        tp (int): Number of true positives
        fp (int): Number of false positives
        fn (int): Number of false negatives

    Returns:
        Union[Tuple[float,float,float], Tuple[np.nan, np.nan, np.nan]]: F1-score
    """
    try:
        precision = get_precision(tp, fp)
        recall = get_recall(tp, (tp+fn))

        f1_score = (2 * precision * recall) / (precision + recall)
        return f1_score, precision, recall
    except ZeroDivisionError:
        return 0, 0, 0


def evaluate(data_dir: str, model: tf.keras.Model, threshold=0.5):
    """Main function for model evaluation. 
    Saves evaluation results to output path.

    Args:
        data_dir (str): Evaluation data directory
        model (tf.keras.Model): Model
        threshold (float, optional): Threshold for prediction confidence. 
        Defaults to 0.5.
    """
    val_path = create_evaluation_dir(cfg.TRAINED_MODEL_PATH)

    input_image_size = cfg.IMAGE_SIZE
    model_fn = model.signatures['serving_default']

    if cfg.ENCODING_TYPE == "winsim":
        window_info = get_window_info(data_dir)
    elif cfg.ENCODING_TYPE == "vdd":
        timestamps_per_trace = get_first_timestamps_vdd(data_dir)

    log_matching = get_log_matching(data_dir)
    drift_info = get_drift_info(data_dir)
    date_info = get_date_info(data_dir)
    traces_per_log = get_traces_per_log(data_dir)

    category_index, _ = utils.get_ex_decoder()

    eval_results = defaultdict(lambda: defaultdict(dict))

    images = get_image_paths(data_dir)

    for image_name, image_path in tqdm(images.items(),
                                       desc="Evaluating model", unit="images"):

        path = os.path.join(image_path, image_name)
        image = utils.load_image(path)
        image = utils.build_inputs_for_object_detection(
            image, input_image_size)
        image = tf.expand_dims(image, axis=0)
        image = tf.cast(image, dtype=tf.uint8)
        result = model_fn(image)

        image_id = int(image_name.split(".")[0])
        log_name = log_matching.loc[log_matching["image_id"] == image_id, "log_name"] \
            .iloc[0]
        min_date, max_date = date_info[log_name]

        scores = result['detection_scores'][0].numpy()

        # For mysterious reasons the TensorFlow bbox format is
        # [y_min, x_min, y_max, x_max]
        bbox_pred = result['detection_boxes'][0].numpy()
        bbox_pred = bbox_pred[scores > threshold]

        y_pred = result['detection_classes'][0].numpy().astype(int)
        y_pred = y_pred[scores > threshold]

        log_info = get_log_info(log_name, drift_info)
        true_change_points = get_true_changepoints_trace_idx(log_info)
        y_true_category = get_true_classes(log_info)
        y_pred_category = get_predicted_classes(y_pred, category_index)

        if cfg.ENCODING_TYPE == "winsim":
            bbox_pred = bbox_pred / cfg.TARGETSIZE \
                * cfg.N_WINDOWS
            log_window_info = window_info[log_name]
            pred_change_points = get_changepoints_trace_idx_winsim(
                bbox_pred, y_pred_category, log_window_info)
        elif cfg.ENCODING_TYPE == "vdd":
            pred_change_points = get_changepoints_trace_idx_vdd(bboxes=bbox_pred,
                                                                y_pred=y_pred_category,
                                                                timestamps_per_trace=timestamps_per_trace[log_name],
                                                                min_date=str_2_date(
                                                                    min_date),
                                                                max_date=str_2_date(max_date))
        for lag_factor in cfg.RELATIVE_LAG:
            metrics, y_pred_sorted, y_true_sorted = get_evaluation_metrics(
                y_true=true_change_points,
                y_pred=pred_change_points,
                y_true_label=y_true_category,
                y_pred_label=y_pred_category,
                factor=lag_factor,
                number_of_traces=traces_per_log[log_name])

            str_lag = f"lag_{lag_factor}"

            eval_results[str_lag][log_name] = \
                {"Detected Changepoints": pred_change_points,
                 "Actual Changepoints": true_change_points,
                 "Predicted Drift Types": y_pred_category,
                 "Actual Drift Types": y_true_category,
                 "F1-Score": metrics["f1"],
                 "Precision": metrics["precision"],
                 "Recall": metrics["recall"],
                 "Average Lag": metrics["lag"],
                 "y_pred_sorted": y_pred_sorted,
                 "y_true_sorted": y_true_sorted
                 }
    eval_results = dict(eval_results)
    for lag_factor in cfg.RELATIVE_LAG:
        str_lag = f"lag_{lag_factor}"
        results_df = save_results(eval_results[str_lag], lag_factor, val_path)
        print_measures(results_df, traces_per_log, lag_factor, val_path)
        plot_classification_report(results_df, val_path, lag_factor)


def get_image_paths(dir: str) -> dict:
    """Get image names and paths in directory.

    Args:
        dir (str): Image directory

    Returns:
        dict: Dictionary, containing image names and paths
    """
    list_of_files = {}
    for dir_path, _, filenames in os.walk(dir):
        for filename in filenames:
            if filename.endswith('.jpg'):
                list_of_files[filename] = dir_path

    return list_of_files


def get_log_matching(data_dir: str) -> pd.DataFrame:
    """Load log matching from file.

    Args:
        data_dir (str): Directory where log matching is stored

    Returns:
        pd.DataFrame: Log matching
    """
    log_matching_path = os.path.join(data_dir, "log_matching.csv")
    assert os.path.isfile(log_matching_path), "No log matching file found"
    log_matching = pd.read_csv(log_matching_path, index_col=0)
    log_matching = log_matching.rename_axis("log_name").reset_index()
    return log_matching


def get_window_info(data_dir: str) -> Union[list, dict]:
    """Load window info from file.

    Args:
        data_dir (str): Directory where window info is stored

    Returns:
        Union[list, dict]: Window info
    """
    window_info_path = os.path.join(data_dir, "window_info.json")
    assert os.path.isfile(window_info_path), "No window info file found"
    file = open_file(window_info_path)
    return json.load(file)


def get_date_info(data_dir: str) -> Union[list, dict]:
    """Load date info from file.

    Args:
        data_dir (str): Directory where date info is stored

    Returns:
        Union[list, dict]: Date info
    """
    date_info_path = os.path.join(data_dir, "date_info.json")
    assert os.path.isfile(date_info_path), "No date info file found"
    file = open_file(date_info_path)
    return json.load(file)


def get_first_timestamps_vdd(data_dir: str) -> Union[list, dict]:
    """Load first timestamps for VDD from file.

    Args:
        data_dir (str): Directory where first timestamps info is stored

    Returns:
        Union[list, dict]: First timestamps info
    """
    first_timestamps_path = os.path.join(data_dir, "first_timestamps.json")
    assert os.path.isfile(first_timestamps_path), "No timestamps file found"
    file = open_file(first_timestamps_path)
    return json.load(file)


def get_traces_per_log(data_dir: str) -> Union[list, dict]:
    """Load traces per log from file.


    Args:
        data_dir (str): Directory where traces per log info is stored

    Returns:
        Union[list, dict]: Number of traces per log
    """
    number_of_traces_path = os.path.join(data_dir, "number_of_traces.json")
    assert os.path.isfile(
        number_of_traces_path), "No number of traces file found"
    file = open_file(number_of_traces_path)
    return json.load(file)


def open_file(path: str):
    """Opens file from path.

    Args:
        path (str): Filepath

    Returns:
        TextIOWrapper: Opened file
    """
    return open(path)


def get_drift_info(data_dir: str) -> pd.DataFrame:
    """Loads drift info from file.

    Args:
        data_dir (str): Directory where drift info is stored

    Returns:
        pd.DataFrame: Drift info
    """
    drift_info_path = os.path.join(data_dir, "drift_info.csv")
    assert os.path.isfile(drift_info_path), "No drift info file found"
    return pd.read_csv(drift_info_path)


def close_file(file):
    """Closes file.

    Args:
        file (TextIOWrapper): File object
    """
    file.close()


def get_changepoints_timestamp_winsim(bboxes: list, y_pred: list,
                                      window_info: dict) -> list:
    """Get changepoints as timestamps for WINSIM encoding.

    Args:
        bboxes (list): List of bboxes
        y_pred (list): List of predictions
        window_info (dict): Window info

    Returns:
        list: List of changepoints as timestamps
    """
    change_points = []
    for i, bbox in enumerate(bboxes):
        if y_pred[i] == "sudden":
            # changepoint is equal to the date of the first trace in the middle
            # window of bbox
            change_point = get_sudden_changepoint_winsim(int(bbox[1]))
        else:
            # changepoint is equal to the date of the first trace in window
            change_point = int(bbox[1])
        change_point_date = dt.datetime.strptime(window_info[str(change_point)][-1],
                                                 "%y-%m-%d").date()
        change_points.append(change_point_date)
    return change_points


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
                change_point = get_sudden_changepoint_winsim(round(bbox[1]))
                change_point_trace_id = (int(window_info[str(change_point)][0]),
                                         int(window_info[str(change_point)][0]))
            else:
                # change start and end is equal to the date of the first trace in window
                # set window id of change start to at least 1 for edge cases
                # set window id of change end to maximum 200 for edge cases
                change_start = (1 if round(bbox[1]) < 1 else round(bbox[1]))
                change_end = (200 if round(bbox[3]) > 200 else round(bbox[3]))
                change_point_trace_id = (int(window_info[str(change_start)][0]),
                                         int(window_info[str(change_end)][0]))
            change_points.append(change_point_trace_id)
        return change_points


def get_changepoints_timestamp_vdd(bboxes: list, y_pred: list,
                                   min_date: dt.date, max_date: dt.date) -> list:
    """Get changepoints as timestamps for VDD encoding. 
    Changepoint is equal to the shift of the relative position starting from the 
    min-date. The time delta is calculated from the relative position, 
    since the x-axis represents the time period.

    Args:
        bboxes (list): List of bboxes
        y_pred (list): List of predictions
        min_date (dt.date): Minimum date in event log
        max_date (dt.date): Maximum date in event log

    Returns:
        list: List of changepoints as timestamps
    """
    change_points = []
    day_delta = max_date - min_date
    for i, bbox in enumerate(bboxes):
        if y_pred[i] == "sudden":
            xmin = get_sudden_changepoint_vdd(int(bbox[1]))
        else:
            xmin = bbox[1]
        relative_xmin = xmin / cfg.TARGETSIZE
        change_point_date = min_date + dt.timedelta(days=int(day_delta.days *
                                                             relative_xmin))
        change_points.append(change_point_date)
    return change_points


def get_changepoints_trace_idx_vdd(bboxes: list, y_pred: list,
                                   timestamps_per_trace: dict,
                                   min_date: dt.date, max_date: dt.date) -> List[tuple]:
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
                xmin = get_sudden_changepoint_vdd(int(bbox[1]))
                relative_xmin = xmin / cfg.TARGETSIZE
                change_point_date = min_date + dt.timedelta(days=int(day_delta.days *
                                                                     relative_xmin))
                closest_trace = get_closest_trace_index(change_point_date,
                                                        timestamps_per_trace)
                change_point_index = (closest_trace, closest_trace)
            else:
                xmin = bbox[1]
                xmax = bbox[3]
                relative_xmin = xmin / cfg.TARGETSIZE
                relative_xmax = xmax / cfg.TARGETSIZE
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


def get_log_info(log_name: str, drift_info: pd.DataFrame) -> pd.DataFrame:
    """Get drift info for event log.

    Args:
        log_name (str): Name of event log
        drift_info (pd.DataFrame): Drift info

    Returns:
        pd.DataFrame: DataFrame, containing drift info for event log
    """
    return drift_info.loc[drift_info["log_name"] == log_name]


def get_true_changepoints_timestamp(log_info: pd.DataFrame) -> list:
    """Get true changepoints as timestamps.

    Args:
        log_info (pd.DataFrame): Drift info for event log

    Returns:
        list: List of true changepoints as timestamps
    """
    change_points_datetime = pd.unique(log_info["change_start"])
    change_points_date = [dt.datetime.strptime(datetime, "%y-%m-%d").date()
                          for datetime in change_points_datetime]
    return change_points_date


def get_true_changepoints_trace_idx(log_info: pd.DataFrame) -> list:
    """Get true changepoints as trace indices.

    Args:
        log_info (pd.DataFrame): Drift info for event log

    Returns:
        list: List of true changepoints as trace indices
    """
    drift_ids = pd.unique(log_info["drift_or_noise_id"])
    change_points_trace_idx = []
    for drift_id in drift_ids:
        drift = log_info.loc[log_info["drift_or_noise_id"] == drift_id]
        if len(drift.index) > 1:
            first_row = drift.iloc[0]["drift_traces_index"]
            first_row = utils.special_string_2_list(first_row)
            last_row = drift.iloc[-1]["drift_traces_index"]
            last_row = utils.special_string_2_list(last_row)
            change_points_trace_idx.append((first_row[0], last_row[-1]))
        else:
            trace_id = drift.iloc[0]["drift_traces_index"]
            trace_id = utils.special_string_2_list(trace_id)
            change_points_trace_idx.append((trace_id[0], trace_id[-1]))
    return change_points_trace_idx


def get_true_classes(log_info: pd.DataFrame) -> list:
    """Get true drift types.

    Args:
        log_info (pd.DataFrame): Drift info for event log

    Returns:
        list: List of true drift types
    """
    drift_ids = pd.unique(log_info["drift_or_noise_id"])
    drift_types = []
    for drift_id in drift_ids:
        drift = log_info.loc[log_info["drift_or_noise_id"] == drift_id]
        drift_type = drift.iloc[0]["drift_type"]
        drift_types.append(drift_type)
    return drift_types


def get_predicted_classes(y_pred: list, category_index: dict) -> list:
    """Get predicted drift types.

    Args:
        y_pred (list): List of prediction values
        category_index (dict): Mapping of drift types to drift IDs

    Returns:
        list: List of predicted drift types
    """
    y_pred_category = []
    for y in y_pred:
        y_pred_category.append(category_index.get(y, {}).get("name"))
    return y_pred_category


def get_day_threshold(min_date: dt.date, max_date: dt.date, day_lag=0.01) -> int:
    """Determine the detection delay in the number of days the prediction is 
    considered a true positive.

    Args:
        min_date (dt.date): First timestamp in event log / observation period.
        max_date (dt.date): Last timestamp in event log / observation period.
        day_lag (float, optional): Allowed detection delay in relation to the time span 
            of the event log. Defaults to 0.01 == 1%.

    Returns:
        int: Lag period in days.
    """
    day_delta = max_date - min_date
    day_threshold = int((day_delta.days * day_lag) / 2)
    return day_threshold


def get_sudden_changepoint_winsim(xmin: int) -> int:
    """Get changepoint for sudden drifts in WINSIM encoding. 

    Args:
        xmin (int): Xmin of predicted bounding box

    Returns:
        int: Trace ID
    """
    if cfg.RESIZE_SUDDEN_BBOX:
        if xmin == 0:
            changepoint = int(cfg.RESIZE_VALUE / 2)
        elif (xmin + cfg.RESIZE_VALUE) == cfg.N_WINDOWS:
            changepoint = int(cfg.N_WINDOWS - (cfg.RESIZE_VALUE / 2))
        else:
            changepoint = xmin + cfg.RESIZE_VALUE
    else:
        if xmin == (cfg.N_WINDOWS - 1):
            changepoint = cfg.N_WINDOWS
        else:
            changepoint = xmin
    return changepoint


def get_sudden_changepoint_vdd(xmin: int) -> int:
    """Get changepoint for sudden drifts in VDD encoding. 

    Args:
        xmin (int): Xmin of predicted bounding box

    Returns:
        int: Trace ID
    """
    factor = 0.02 * cfg.TARGETSIZE

    changepoint = int(xmin + factor)
    if cfg.RESIZE_SUDDEN_BBOX:
        if xmin == 0:
            changepoint = int(factor / 2)
        elif int(xmin + factor) >= cfg.TARGETSIZE:
            changepoint = int(cfg.TARGETSIZE - (factor / 2))
        else:
            changepoint = int(xmin + factor)
    else:
        if xmin == (cfg.TARGETSIZE - 10):
            changepoint = cfg.TARGETSIZE
        else:
            changepoint = xmin
    return changepoint


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
                              nearest(timestamps_df["timestamp"].to_list(),
                                      drift_moment_date)].index[0]

    return int(timestamps_df.iloc[index]["trace_id"])


def save_results(results: dict, lag_factor: float, val_path: str) -> pd.DataFrame:
    """Saves evaluation results to output path.

    Args:
        results (dict): Evaluation measures
        lag_factor (float): Relative lag
        val_path (str): Evaluation directory
    
    Returns:
        pd.DataFrame: DataFrame, containing evaluation measures
    """
    results_df = pd.DataFrame.from_dict(results, orient="index")
    save_path = os.path.join(val_path,
                             f"evaluation_results_{cfg.EVAL_MODE}_{lag_factor}_lag.csv")
    results_df.to_csv(save_path, sep=",")
    return results_df


def str_2_date(date: str) -> dt.date:
    """Convert string to date object.

    Args:
        date (str): String of format "%m-%d-%Y"

    Returns:
        dt.date: Date object of format "%m-%d-%Y"
    """
    return dt.datetime.strptime(date, "%m-%d-%Y").date()


def nearest(items: list, pivot):
    """Return nearest item from list.
    Source:
    https://stackoverflow.com/questions/32237862/find-the-closest-date-to-a-given-date
    Args:
        items (list): List of search items
        pivot (any): Search item

    Returns:
        any: Element that is closest to pivot
    """
    return min([i for i in items if i <= pivot], key=lambda x: abs(x - pivot))


def print_measures(results: pd.DataFrame, num_traces: dict,
                   lag_factor: float, val_path: str):
    """Generate evaluation measure overview. 
    Calculates total average F1-score and total average lag.

    Args:
        results (pd.DataFrame): Evaluation results
        num_traces (dict): Number of traces per log
        lag_factor (float): Relative lag
        val_path (str): Evaluation directory
    """
    num_events = len(num_traces.keys())

    f1_values = np.nansum(results["F1-Score"])
    lag_values = np.nansum(results["Average Lag"])

    total_average_f1 = f1_values / num_events
    total_average_lag = lag_values / num_events
    average_length = int(sum(num_traces.values()) / num_events)

    text_path = os.path.join(val_path,
                             f"evaluation_report_{cfg.EVAL_MODE}_{lag_factor}_lag.txt")

    with open(text_path, "w") as f:
        f.write(
            "---------------------------------------------------------------------\n")
        f.write("\n")
        f.write("EVALUATION REPORT:")
        f.write("\n")
        f.write(f"PREDICTION THRESHOLD: {cfg.EVAL_THRESHOLD}")
        f.write("\n")
        f.write(
            "---------------------------------------------------------------------\n")
        f.write(f"In total {num_events} were evaluated, with an average trace length of\
                {average_length}\n")
        f.write("\n")
        f.write(
            "---------------------------------------------------------------------\n")
        f.write("\n")
        f.write(
            f"The average f1 score for all evaluated logs is: {total_average_f1}.\n")
        f.write(
            "---------------------------------------------------------------------\n")
        f.write("\n")
        f.write(
            f"The average lag for all evaluated logs is: {total_average_lag}.\n")
        f.write(
            "---------------------------------------------------------------------\n")


def create_evaluation_dir(path: str) -> str:
    """Create directory for evaluation files.
    If dir already exists, skip creation.

    Args:
        path (str): Model path

    Returns:
        str: Path of evaluation directory
    """
    val_path = os.path.join(path, "evaluation",
                            f"{cfg.EVAL_MODE}",
                            f"threshold_{cfg.EVAL_THRESHOLD}")
    if not os.path.isdir(val_path):
        os.makedirs(val_path)
    return val_path


def plot_classification_report(results: pd.DataFrame, path: str, lag_factor: float):
    """Plot classification report.

    Args:
        results (pd.DataFrame): Evaluation results
        path (str): Save path
        lag_factor (float): Lag factor used
    """
    y_trues = results["y_true_sorted"].to_numpy()
    y_preds = results["y_pred_sorted"].to_numpy()

    y_trues_flat = [elem for sub in y_trues for elem in sub]
    y_preds_flat = [elem for sub in y_preds for elem in sub]

        
    results = get_FP_TP_per_class(y_preds_flat, y_trues_flat)
    
    c_r = {}
    
    average_precision = 0
    average_recall = 0
    average_f1 = 0
    
    for k in results.keys():
        f1, precision, recall = get_f1_score_per_class(tp=results[k]["TP"], 
                                                       fp=results[k]["FP"], 
                                                       fn=results[k]["FN"])
        c_r[k] = {"precision": precision, 
                  "recall": recall, 
                  "f1-score": f1}
        average_precision += precision
        average_recall += recall
        average_f1 += f1
        
    average_precision = average_precision / len(c_r)
    average_recall = average_recall / len(c_r)
    average_f1 = average_f1 / len(c_r)   
    c_r["average"] = {"precision": average_precision, 
                      "recall": average_recall, 
                      "f1-score": average_f1}

    # exclude support from clf report with iloc
    clf_r = sns.heatmap(pd.DataFrame(
        c_r).T, cmap="YlGn", annot=True, cbar=False)
    clf_r_fig = clf_r.get_figure()
    save_path = os.path.join(
        path, f"classification_report_{cfg.EVAL_MODE}_{lag_factor}_lag.png")
    clf_r_fig.tight_layout()
    clf_r_fig.savefig(save_path)
    print(f"Classification report is saved at: {save_path}")
    plt.close(clf_r_fig)


def call_pro_drift(log_path: str, pro_drift_dir: str,
                   window_size: int) -> Tuple[bytes, bytes]:
    """Call ProDrift for evaluation purposes.

    Args:
        log_path (str): Logs to evaluate
        pro_drift_dir (str): Directory of ProDrift distribution
        window_size (int): Window size for ProDrift

    Returns:
        Tuple[bytes, bytes]: Process output and error log
    """
    env = dict(os.environ)
    env['JAVA_OPTS'] = 'foo'

    cmd = f"java -jar ProDrift2.5.jar -fp {log_path} \
        -ddm runs -ws {window_size} -gradual"

    if cfg.WINDOWS_SYSTEM:
        p = subprocess.Popen(cmd,
                             stdout=subprocess.PIPE,
                             shell=True,
                             cwd=pro_drift_dir,
                             env=env)
        try:
            outs, errs = p.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            p.kill()
            outs, errs = p.communicate()
    else:
        p = subprocess.Popen(cmd,
                             stdout=subprocess.PIPE,
                             cwd=pro_drift_dir,
                             shell=True,
                             env=env,
                             preexec_fn=os.setsid)
        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
    return outs, errs


def call_vdd(log_dir: str, vdd_dir: str):
    """Call VDD for evaluation purposes.

    Args:
        log_dir (str): Logs to evaluate
        vdd_dir (str): Directory of VDD distribution
    """
    vdd_data_input_dir = os.path.join(vdd_dir, "data", "data_input")

    if vdd_data_input_dir != log_dir:
        print("Copying log files to VDD data directory.")
        shutil.copytree(log_dir, vdd_data_input_dir)

    files = []
    for (dirpath, dirnames, filenames) in os.walk(vdd_data_input_dir):
        files.extend(filenames)
        break

    filenames = [elem.split(".")[0] for elem in files]

    for filename in filenames:
        cmd = f"python -m src.scenario_1 -logName {filename} \
            -subL 100 -sliBy 50 -driftAll"
        p = subprocess.Popen(cmd,
                             stdout=subprocess.PIPE,
                             cwd="/ceph/jkoessle/Process-Drift-Visualization-With-Declare",
                             shell=True)
        try:
            outs, errs = p.communicate()
        except subprocess.TimeoutExpired:
            p.kill()
            outs, errs = p.communicate()


def excel_2_csv(excel_path: str, csv_path: str):
    """Convert Excel file to CSV format.

    Args:
        excel_path (str): Path of Excel file
        csv_path (str): Output path for CSV file
    """
    excel_file = pd.read_excel(excel_path)
    excel_file.to_csv(csv_path, index=None, header=True,
                      sep=",", encoding="utf-8")


def preprocess_pro_drift_results(results_path: str):
    """Transform changepoints into tuples. 
    Caution: Overwrites given CSV file. 
    Gradual drifts and their change points must be handled manually after this process, 
    since their change points are unfortunately split into two tuples.

    Args:
        results_path (str): CSV filepath of results file
    """
    results_df = pd.read_csv(results_path)
    trace_idx = results_df["change_trace_idx"].to_numpy()
    tuples_list = []
    for elem in trace_idx:
        if str(elem) == "nan":
            tuples_list.append(np.NaN)
            continue
        integers = [int(s) for s in re.findall(r'\b\d+\b', elem)]
        for i, integer in enumerate(integers):
            integers[i] = (integer, integer)
        tuples_list.append(integers)
    results_df["change_trace_idx"] = tuples_list

    results_df.to_csv(results_path, index=None, header=True,
                      sep=",", encoding="utf-8")


def evaluate_pro_drift_results(results_file_path: str, data_dir: str):
    """Evaluate ProDrift with score measures.

    Args:
        results_file_path (str): Results file
        data_dir (str): Test data dir
    """
    results_df = pd.read_csv(results_file_path, sep=",", encoding="utf-8")
    results_df["change_trace_idx"] = results_df.loc[
        results_df["change_trace_idx"].notnull(), "change_trace_idx"
        ].apply(lambda x: ast.literal_eval(x))
    drift_info = get_drift_info(data_dir)
    traces_per_log = get_traces_per_log(data_dir)
    eval_results = defaultdict(lambda: defaultdict(dict))
    log_names = results_df["log_name"]

    val_path = os.path.abspath(os.path.join("evaluation_results", "ProDrift"))

    for log_name in log_names:
        log_info = get_log_info(log_name, drift_info)
        pred_change_points = results_df["change_trace_idx"].loc[
            results_df["log_name"] == log_name].tolist()
        if str(pred_change_points[0]) == "nan":
            pred_change_points = []
        else:
            pred_change_points = [
                elem for sub in pred_change_points for elem in sub]

        true_change_points = get_true_changepoints_trace_idx(log_info)

        y_true_category = get_true_classes(log_info)
        y_pred_category = results_df["drift_types"].loc[
            results_df["log_name"] == log_name].to_numpy().tolist()
        if str(y_pred_category[0]) == "nan":
            y_pred_category = []
        else:
            y_pred_category = [x.strip()
                               for x in y_pred_category[0].split(",")]

        for lag_factor in cfg.RELATIVE_LAG:
            metrics, y_pred_sorted, y_true_sorted = get_evaluation_metrics(
                y_true=true_change_points,
                y_pred=pred_change_points,
                y_true_label=y_true_category,
                y_pred_label=y_pred_category,
                factor=lag_factor,
                number_of_traces=traces_per_log[log_name])

            str_lag = f"lag_{lag_factor}"

            eval_results[str_lag][log_name] = \
                {"Detected Changepoints": pred_change_points,
                 "Actual Changepoints": true_change_points,
                 "Predicted Drift Types": y_pred_category,
                 "Actual Drift Types": y_true_category,
                 "F1-Score": metrics["f1"],
                 "Precision": metrics["precision"],
                 "Recall": metrics["recall"],
                 "Average Lag": metrics["lag"],
                 "y_pred_sorted": y_pred_sorted,
                 "y_true_sorted": y_true_sorted
                 }
    eval_results = dict(eval_results)
    for lag_factor in cfg.RELATIVE_LAG:
        str_lag = f"lag_{lag_factor}"
        results_df = save_results(eval_results[str_lag], lag_factor, val_path)
        print_measures(results_df, traces_per_log, lag_factor, val_path)
        plot_classification_report(results_df, val_path, lag_factor)
