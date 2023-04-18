import tensorflow as tf
import numpy as np
import os
import pandas as pd
import cnn_image_detection.utils.config as cfg


def get_event_log_paths():
    list_of_files = {}
    for dir_path, dir_names, filenames in os.walk(cfg.DEFAULT_LOG_DIR):
        for filename in filenames:
            if filename.endswith('.xes'):
                list_of_files[filename] = dir_path

    assert len(list_of_files) > 0, f"{cfg.DEFAULT_LOG_DIR} is empty"

    return list_of_files


def get_drift_coordinates(change_idx):

    if len(change_idx) > 1:
        xmin = ymin = change_idx[0]
        xmax = ymax = change_idx[1]
    else:
        xmin = ymin = xmax = ymax = change_idx[0]

    return np.array((xmin, ymin, xmax, ymax))


def update_trace_indices(df: pd.DataFrame, log_name, borders):

    df_filtered = df.loc[df["log_name"] == log_name]

    df_filtered.loc[:, "change_trace_index"] = df_filtered.loc[:, "change_trace_index"]\
        .map(lambda x: df_trace_index_2_window_id(borders, x))
    df_filtered.loc[:, "change_trace_index"] = df_filtered.loc[:, "change_trace_index"]\
        .map(get_drift_coordinates)

    df.update(df_filtered)

    return df


def df_trace_index_2_window_id(borders, trace_ids):
    result_set = []

    if len(trace_ids) > 1:
        drift_start = trace_ids[0]
        drift_end = trace_ids[1]

        for i, elem in enumerate(borders):
            left, right = elem
            if is_in_window(left, drift_start, right):
                result_set.insert(0, i)
            elif is_in_window(left, drift_end, right):
                result_set.append(i)
            if len(result_set) == len(trace_ids):
                break
    else:
        drift_start = trace_ids[0]
        for i, elem in enumerate(borders):
            left, right = elem
            if is_in_window(left, drift_start, right):
                result_set.append(i)
                break

    return result_set


def is_in_window(left, x, right):
    return left <= x <= right


def df_is_in_window(left, x, right):
    result = []
    if len(x) > 1:
        for elem in x:
            r = left <= elem <= right
            result.append(r)
    else:
        result.append(left <= x[0] <= right)
    return result


def special_string_2_list(s):
    return list(map(int, s.translate({ord(i): None for i in "[]"}).split(",")))


def bbox_corner_to_center(box):
    xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]
    x = (xmin + xmax) / 2
    y = (ymin + ymax) / 2
    width = xmax - xmin
    height = ymax - ymin
    boxes = tf.stack((x, y, width, height), axis=-1)
    return boxes


def bbox_center_to_corner(box):
    x, y, width, height = box[0], box[1], box[2], box[3]
    xmin = x - 0.5 * width
    ymin = y - 0.5 * height
    xmax = x + 0.5 * width
    ymax = y + 0.5 * height
    boxes = tf.stack((xmin, ymin, xmax, ymax), axis=-1)
    return boxes


# TODO
def generate_annotations(drift_info):
    pass


def get_drift_info() -> pd.DataFrame:
    path = os.path.join(cfg.DEFAULT_LOG_DIR, "drift_info.csv")
    return pd.read_csv(path)


def extract_drift_information() -> pd.DataFrame:

    pd_df = get_drift_info()

    indices = pd_df[pd_df["drift_sub_attribute"] == "change_trace_index"]
    indices = indices[["log_name", "drift_or_noise_id",
                       "drift_attribute", "value"]]
    indices = indices.rename(columns={"value": "change_trace_index"})

    change_types = pd_df[pd_df["drift_sub_attribute"] == "change_type"]
    change_types = change_types[[
        "log_name", "drift_or_noise_id", "drift_attribute", "value"]]
    change_types = change_types.rename(columns={"value": "change_type"})

    drift_types = pd_df[pd_df["drift_attribute"] == "drift_type"]
    drift_types = drift_types[["log_name", "drift_or_noise_id", "value"]]
    drift_types = drift_types.rename(columns={"value": "drift_type"})

    drift_info = drift_types.merge((indices.merge(change_types, on=[
                                   "log_name", "drift_or_noise_id", "drift_attribute"])
    ), on=["log_name", "drift_or_noise_id"])

    drift_info["change_trace_index"] = drift_info["change_trace_index"].map(
        special_string_2_list)

    return drift_info
