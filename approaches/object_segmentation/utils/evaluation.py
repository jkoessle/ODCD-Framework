import os
import json
import pandas as pd
import tensorflow as tf

import datetime as dt

from . import config as cfg
from . import utilities as utils


def get_f1_score():
    pass


def get_accuracy():
    pass


def get_recall():
    pass


def get_tp_fp():
    pass


def get_average_lag():
    pass


def get_evaluation_results(log_dir, eval_dir, model, threshold=0.5):

    data = tf.data.TFRecordDataset(eval_dir)
    input_image_size = cfg.IMAGE_SIZE
    model_fn = model.signatures['serving_default']
    
    if cfg.ENCODING_TYPE == "winsim":
            window_info = get_window_info(log_dir)
    elif cfg.ENCODING_TYPE == "vdd":
            timestamps_per_trace = get_first_timestamps_vdd(log_dir)
            
    log_matching = get_log_matching(log_dir)
    drift_info = get_drift_info(log_dir)
    date_info = get_date_info(log_dir)

    category_index, tf_ex_decoder = utils.get_ex_decoder()

    eval_results = {}

    for i, tfr_tensor in enumerate(data):
        decoded_tensor = tf_ex_decoder.decode(tfr_tensor)
        image = utils.build_inputs_for_object_detection(
            decoded_tensor['image'], input_image_size)
        image = tf.expand_dims(image, axis=0)
        image = tf.cast(image, dtype=tf.uint8)
        # image_np = image[0].numpy()
        result = model_fn(image)

        # result = np.where(result['detection_scores'][0].numpy() > threshold)

        # check if that is correct
        image_id = result["image_id"]
        log_name = log_matching.loc[log_matching["image_id"] == image_id, "log_name"] \
            .iloc[0]
        min_date, max_date = date_info[log_name]

        scores = result['detection_scores'][0].numpy()

        bbox_pred = result['detection_boxes'][0].numpy()
        bbox_pred = bbox_pred[scores > threshold]

        y_pred = result['detection_classes'][0].numpy().astype(int)
        y_pred = y_pred[bbox_pred]

        log_info = get_log_info(log_name, drift_info)
        true_change_points = get_true_changepoints_trace_idx(log_info)
        y_true_category = get_true_classes(log_info)
        y_pred_category = get_predicted_classes(y_pred, category_index)

        if cfg.ENCODING_TYPE == "winsim":
            # window_info = get_window_info(log_dir)
            bbox_pred = bbox_pred / cfg.TARGETSIZE \
                * cfg.N_WINDOWS
            log_window_info = window_info[log_name]
            pred_change_points = get_changepoints_trace_idx_winsim(
                bbox_pred, y_pred_category, log_window_info)
        elif cfg.ENCODING_TYPE == "vdd":
            pred_change_points = get_changepoints_trace_idx_vdd(bboxes=bbox_pred,
                                                      y_pred=y_pred_category,
                                                      timestamps_per_trace=
                                                      timestamps_per_trace[log_name],
                                                      min_date=min_date.date(),
                                                      max_date=max_date.date())
            

        day_threshold = get_day_threshold(min_date=min_date.date(),
                                          max_date=max_date.date())
        # TODO
        f1_score = get_f1_score()
        lag_score = get_average_lag()

        eval_results[log_name] = {"Detected Changepoint Dates": pred_change_points,
                                  "Actual Changepoint Dates": true_change_points,
                                  "Predicted Drift Types": y_pred_category,
                                  "Actual Drift Types": y_true_category,
                                  "F1-Score": f1_score,
                                  "Average Lag": lag_score
                                  }

    if cfg.ENCODING_TYPE == "winsim":
        close_file(window_info)
    elif cfg.ENCODING_TYPE == "vdd":
        close_file(timestamps_per_trace)

    close_file(date_info)

    return eval_results


def get_log_matching(log_dir):
    log_matching_path = os.path.join(log_dir, "log_matching.csv")
    assert os.path.isfile(log_matching_path), "No log matching file found"
    log_matching = pd.read_csv(log_matching_path)
    log_matching = log_matching.rename_axis("log_name").reset_index()
    return log_matching


def get_window_info(log_dir):
    window_info_path = os.path.join(log_dir, "window_info.json")
    assert os.path.isfile(window_info_path), "No window info file found"
    return json.load(window_info_path)


def get_date_info(log_dir):
    date_info_path = os.path.join(log_dir, "date_info.json")
    assert os.path.isfile(date_info_path), "No date info file found"
    return json.load(date_info_path)


def get_first_timestamps_vdd(log_dir):
    first_timestamps_path = os.path.join(log_dir, "first_timestamps.json")
    assert os.path.isfile(first_timestamps_path), "No timestamps file found"
    return json.load(first_timestamps_path)


def get_drift_info(log_dir):
    drift_info_path = os.path.join(log_dir, "drift_info.csv")
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


def get_changepoints_trace_idx_winsim(bboxes, y_pred, window_info) -> list:
    change_points = []
    for i, bbox in enumerate(bboxes):
        if y_pred[i] == "sudden":
            # changepoint is equal to the date of the first trace in the middle
            # window of bbox
            change_point = get_sudden_changepoint_winsim(int(bbox[0]))
        else:
            # changepoint is equal to the date of the first trace in window
            change_point = int(bbox[0])
        change_point_trace_id = int(window_info[str(change_point)][0])
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


def get_changepoints_trace_idx_vdd(bboxes, y_pred, timestamps_per_trace: pd.DataFrame,
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
        change_point_index = get_closest_trace_index(change_point_date, 
                                                     timestamps_per_trace)
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
    change_points_trace_indices = list(log_info["drift_traces_index"])
    change_points_trace_idx = []
    for elem in change_points_trace_indices:
        elem = utils.special_string_2_list(elem)
        change_points_trace_idx.append(elem[0])
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


def get_day_threshold(min_date: dt.date, max_date: dt.date) -> int:
    day_delta = max_date - min_date
    day_threshold = int(day_delta.days / 2)
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
                            timetamps_per_trace: dict):
    timestamps_df = pd.DataFrame.from_dict(timetamps_per_trace, 
                                           orient="index", 
                                           columns=["timestamp"])
    timestamps_df = timestamps_df.rename_axis("trace_id").reset_index()
    timestamps_df["timestamp"] = timestamps_df["timestamp"].apply(lambda _: 
        dt.datetime.strptime(_,"%m-%d-%Y").date())
    index = timestamps_df["timestamp"].searchsorted(drift_moment_date)
    return timestamps_df.iloc[index]["trace_id"] 
    
    
def evaluate():
    pass
