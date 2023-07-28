import os
import json
import pandas as pd
import tensorflow as tf
import numpy as np

import datetime as dt

from typing import List, Tuple, Union
from . import config as cfg
from . import utilities as utils
from . import cdrift_evaluation as cdrift


def get_evaluation_metrics(y_true: list, y_pred: list,
                           factor: float, number_of_traces: int) -> dict:
    """Get all relevant evaluation metrics.

    Args:
        y_true (list): List of groundtruth values
        y_pred (list): List of prediction values
        factor (float): Factor for relative lag
        number_of_traces (int): Number of traces for event log

    Returns:
        dict: Dict, containing evaluation measures
    """
    lag = int(factor * number_of_traces)

    tp_fp, assignments = cdrift.getTP_FP(detected=y_pred,
                                         known=y_true,
                                         lag=lag)
    tp, fp = tp_fp

    f1, precision, recall = get_f1_score(tp, fp, len(y_true))
    average_lag = get_average_lag(assignments)

    metrics = {"f1": f1,
               "precision": precision,
               "recall": recall,
               "lag": average_lag}
    return metrics


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


def get_f1_score(tp: int, fp: int, y_true_size: int) \
    -> Union[Tuple[float, float, float],Tuple[np.nan, np.nan, np.nan]]:
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


def evaluate(data_dir: str, model: tf.keras.Model, threshold=0.5):
    """Main function for model evaluation. 
    Saves evaluation results to output path.

    Args:
        data_dir (str): Evaluation data directory
        model (tf.keras.Model): Model
        threshold (float, optional): Threshold for prediction confidence. 
        Defaults to 0.5.
    """
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

    category_index, tf_ex_decoder = utils.get_ex_decoder()

    eval_results = {}

    images = get_image_paths(data_dir)

    for image_name, image_path in images.items():

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

        metrics = get_evaluation_metrics(y_true=true_change_points,
                                         y_pred=pred_change_points,
                                         factor=cfg.RELATIVE_LAG,
                                         number_of_traces=traces_per_log[log_name])

        eval_results[log_name] = {"Detected Changepoints": pred_change_points,
                                  "Actual Changepoints": true_change_points,
                                  "Predicted Drift Types": y_pred_category,
                                  "Actual Drift Types": y_true_category,
                                  "F1-Score": metrics["f1"],
                                  "Precision": metrics["precision"],
                                  "Recall": metrics["recall"],
                                  "Average Lag": metrics["lag"]
                                  }

    save_results(eval_results)


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
    return open(path)


def get_drift_info(data_dir: str) -> pd.DataFrame:
    """Load drift info from file.

    Args:
        data_dir (str): Directory where drift info is stored

    Returns:
        pd.DataFrame: Drift info
    """
    drift_info_path = os.path.join(data_dir, "drift_info.csv")
    assert os.path.isfile(drift_info_path), "No drift info file found"
    return pd.read_csv(drift_info_path)


def close_file(file):
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
            change_point = get_sudden_changepoint_winsim(int(bbox[0]))
        else:
            # changepoint is equal to the date of the first trace in window
            change_point = int(bbox[0])
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
    for i, bbox in enumerate(bboxes):
        if y_pred[i] == "sudden":
            # changepoint is equal to the date of the first trace in the middle
            # window of bbox
            change_point = get_sudden_changepoint_winsim(int(bbox[0]))
            change_point_trace_id = (int(window_info[str(change_point)][0]),
                                     int(window_info[str(change_point)][0]))
        else:
            # change start and end is equal to the date of the first trace in window
            change_start = int(bbox[0])
            change_end = int(bbox[0] + bbox[2])
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
            xmin = get_sudden_changepoint_vdd(int(bbox[0]))
        else:
            xmin = bbox[0]
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
    day_delta = max_date - min_date
    for i, bbox in enumerate(bboxes):
        if y_pred[i] == "sudden":
            xmin = get_sudden_changepoint_vdd(int(bbox[0]))
            relative_xmin = xmin / cfg.TARGETSIZE
            change_point_date = min_date + dt.timedelta(days=int(day_delta.days *
                                                                 relative_xmin))
            closest_trace = get_closest_trace_index(change_point_date,
                                                    timestamps_per_trace)
            change_point_index = (closest_trace, closest_trace)
        else:
            xmin = bbox[0]
            xmax = bbox[0] + bbox[2]
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
            change_points_trace_idx.append((trace_id[0], trace_id[0]))
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


def save_results(results: dict):
    """Saves evaluation results to output path.

    Args:
        results (dict): Evaluation measures
    """
    results_df = pd.DataFrame.from_dict(results, orient="index")
    save_path = os.path.join(cfg.TRAINED_MODEL_PATH, "evaluation_results.csv")
    results_df.to_csv(save_path, sep=",")


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
