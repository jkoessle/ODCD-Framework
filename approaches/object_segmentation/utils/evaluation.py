import os
import json
import pandas as pd
import tensorflow as tf
import numpy as np

import datetime as dt

from typing import List, Tuple
from . import config as cfg
from . import utilities as utils
from . import cdrift_evaluation as cdrift


def get_evaluation_metrics(y_true: list, y_pred: list) -> dict:

    tp_fp, assignments = cdrift.getTP_FP(detected=y_pred,
                                         known=y_true,
                                         lag=200)
    tp, fp = tp_fp

    f1, precision, recall = get_f1_score(tp, fp, len(y_true))
    average_lag = get_average_lag(assignments)

    metrics = {"f1": f1,
               "precision": precision,
               "recall": recall,
               "lag": average_lag}
    return metrics


def get_precision(tp, fp):
    return tp / (tp+fp)


def get_recall(tp, y_true_size):
    return tp / y_true_size


def get_f1_score(tp, fp, y_true_size):
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


def evaluate(data_dir, model, threshold=0.5):

    # data = tf.data.TFRecordDataset(eval_dir)
    input_image_size = cfg.IMAGE_SIZE
    model_fn = model.signatures['serving_default']

    if cfg.ENCODING_TYPE == "winsim":
        window_info = get_window_info(data_dir)
    elif cfg.ENCODING_TYPE == "vdd":
        timestamps_per_trace = get_first_timestamps_vdd(data_dir)

    log_matching = get_log_matching(data_dir)
    drift_info = get_drift_info(data_dir)
    date_info = get_date_info(data_dir)

    category_index, tf_ex_decoder = utils.get_ex_decoder()

    eval_results = {}

    images = get_image_paths(data_dir)

    for image_name, image_path in images.items():

        # for i, tfr_tensor in enumerate(data):
        # decoded_tensor = tf_ex_decoder.decode(tfr_tensor)
        # image = utils.build_inputs_for_object_detection(
        #     decoded_tensor['image'], input_image_size)
        path = os.path.join(image_path, image_name)
        image = utils.load_image(path)
        image = utils.build_inputs_for_object_detection(
            image, input_image_size)
        image = tf.expand_dims(image, axis=0)
        image = tf.cast(image, dtype=tf.uint8)
        # image_np = image[0].numpy()
        result = model_fn(image)

        # result = np.where(result['detection_scores'][0].numpy() > threshold)

        # check if that is correct
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
            # window_info = get_window_info(data_dir)
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
                                         y_pred=pred_change_points)

        eval_results[log_name] = {"Detected Changepoints": pred_change_points,
                                  "Actual Changepoints": true_change_points,
                                  "Predicted Drift Types": y_pred_category,
                                  "Actual Drift Types": y_true_category,
                                  "F1-Score": metrics["f1"],
                                  "Precision": metrics["precision"],
                                  "Recall": metrics["recall"],
                                  "Average Lag": metrics["lag"]
                                  }

    # if cfg.ENCODING_TYPE == "winsim":
    #     close_file(window_info)
    # elif cfg.ENCODING_TYPE == "vdd":
    #     close_file(timestamps_per_trace)

    # close_file(date_info)

    save_results(eval_results)


def get_image_paths(dir):
    list_of_files = {}
    for dir_path, _, filenames in os.walk(dir):
        for filename in filenames:
            if filename.endswith('.jpg'):
                list_of_files[filename] = dir_path

    return list_of_files


def get_log_matching(data_dir):
    log_matching_path = os.path.join(data_dir, "log_matching.csv")
    assert os.path.isfile(log_matching_path), "No log matching file found"
    log_matching = pd.read_csv(log_matching_path, index_col=0)
    log_matching = log_matching.rename_axis("log_name").reset_index()
    return log_matching


def get_window_info(data_dir):
    window_info_path = os.path.join(data_dir, "window_info.json")
    assert os.path.isfile(window_info_path), "No window info file found"
    file = open_file(window_info_path)
    return json.load(file)


def get_date_info(data_dir):
    date_info_path = os.path.join(data_dir, "date_info.json")
    assert os.path.isfile(date_info_path), "No date info file found"
    file = open_file(date_info_path)
    return json.load(file)


def get_first_timestamps_vdd(data_dir):
    first_timestamps_path = os.path.join(data_dir, "first_timestamps.json")
    assert os.path.isfile(first_timestamps_path), "No timestamps file found"
    file = open_file(first_timestamps_path)
    return json.load(file)


def open_file(path):
    return open(path)


def get_drift_info(data_dir):
    drift_info_path = os.path.join(data_dir, "drift_info.csv")
    assert os.path.isfile(drift_info_path), "No drift info file found"
    return pd.read_csv(drift_info_path)


def close_file(file):
    file.close()


def get_changepoints_timestamp_winsim(bboxes, y_pred, window_info) -> list:
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


def get_changepoints_trace_idx_winsim(bboxes, y_pred, window_info) -> List[tuple]:
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


def get_changepoints_timestamp_vdd(bboxes, y_pred,
                                   min_date: dt.date, max_date: dt.date) -> list:
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


def get_changepoints_trace_idx_vdd(bboxes, y_pred, timestamps_per_trace: dict,
                                   min_date: dt.date, max_date: dt.date) -> List[tuple]:
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
    return drift_info.loc[drift_info["log_name"] == log_name]


def get_true_changepoints_timestamp(log_info: pd.DataFrame) -> list:
    change_points_datetime = pd.unique(log_info["change_start"])
    change_points_date = [dt.datetime.strptime(datetime, "%y-%m-%d").date()
                          for datetime in change_points_datetime]
    return change_points_date


def get_true_changepoints_trace_idx(log_info: pd.DataFrame) -> list:
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
    drift_ids = pd.unique(log_info["drift_or_noise_id"])
    drift_types = []
    for drift_id in drift_ids:
        drift = log_info.loc[log_info["drift_or_noise_id"] == drift_id]
        drift_type = drift.iloc[0]["drift_type"]
        drift_types.append(drift_type)
    return drift_types


def get_predicted_classes(y_pred, category_index: dict) -> list:
    y_pred_category = []
    for y in y_pred:
        # y_pred_str.append(category_index[y]["name"])
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
    results_df = pd.DataFrame.from_dict(results, orient="index")
    save_path = os.path.join(cfg.TRAINED_MODEL_PATH, "evaluation_results.csv")
    results_df.to_csv(save_path, sep=",")


def str_2_date(date: str):
    return dt.datetime.strptime(date, "%m-%d-%Y").date()


def nearest(items: list, pivot):
    """Return nearest item from list
    Source:
    https://stackoverflow.com/questions/32237862/find-the-closest-date-to-a-given-date
    Args:
        items (list): List of search items
        pivot (any): Search item

    Returns:
        any: Element that is closest to pivot
    """
    return min([i for i in items if i <= pivot], key=lambda x: abs(x - pivot))
