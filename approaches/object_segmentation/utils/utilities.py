import os
import json
import datetime as dt
import pytz
import subprocess
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pm4py as pm
from . import config as cfg

from PIL import Image
from tqdm import tqdm
from official.vision.dataloaders.tf_example_decoder import TfExampleDecoder
from official.vision.utils.object_detection import visualization_utils
from official.core import exp_factory
from official.core.config_definitions import ExperimentConfig
from typing import Union, Tuple
from pm4py.objects.log.obj import EventLog
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.filtering.log.attributes import attributes_filter
from official.vision.ops.preprocess_ops import resize_and_crop_image
from pm4py import discover_dfg_typed


def get_event_log_paths(dir: str) -> dict:
    """Get event logs for directory.

    Args:
        dir (str): Log directory

    Returns:
        dict: Dictionary, containing names and paths of event logs
    """
    list_of_files = {}
    for dir_path, dir_names, filenames in os.walk(dir):
        for filename in filenames:
            if filename.endswith('.xes'):
                list_of_files[filename] = dir_path

    assert len(list_of_files) > 0, f"{dir} is empty"

    return list_of_files


def import_event_log(path: str, name: str) -> EventLog:
    """Read event log from path and return EventLog object.

    Args:
        path (str): Log path
        name (str): Log name

    Returns:
        EventLog: Imported event log
    """
    variant = xes_importer.Variants.ITERPARSE
    parameters = {variant.value.Parameters.TIMESTAMP_SORT: True,
                  variant.value.Parameters.SHOW_PROGRESS_BAR: False}
    event_log = xes_importer.apply(os.path.join(
        path, name), variant=variant, parameters=parameters)

    return event_log


def filter_complete_events(log: EventLog) -> EventLog:
    """Filter event log on complete events.

    Args:
        log (EventLog): Imported event log

    Returns:
        EventLog: Filtered event log
    """
    try:
        filtered_log = attributes_filter.apply_events(log, ["complete", "COMPLETE"], 
            parameters={
            attributes_filter.Parameters.ATTRIBUTE_KEY: "lifecycle:transition",
            attributes_filter.Parameters.POSITIVE: True})
    except Exception:
        filtered_log = log

    return filtered_log


def filter_complete_events_uppercase(log: EventLog) -> EventLog:
    """Filter event log on complete events.

    Args:
        log (EventLog): Imported event log

    Returns:
        EventLog: Filtered event log
    """
    try:
        filtered_log = attributes_filter.apply_events(log, ["COMPLETE"], parameters={
            attributes_filter.Parameters.ATTRIBUTE_KEY: "lifecycle:transition",
            attributes_filter.Parameters.POSITIVE: True})
    except Exception:
        filtered_log = log

    return filtered_log


def get_drift_coordinates(change_idx: list) -> list:
    """Get drift bbox.

    Args:
        change_idx (list): List of change indices

    Returns:
        list: Bbox of drift
    """
    assert len(change_idx) > 0, "Error in border mapping"

    if len(change_idx) > 1:
        xmin = ymin = change_idx[0]
        xmax = ymax = change_idx[1]
    elif len(change_idx) == 1:
        xmin = ymin = xmax = ymax = change_idx[0]

    return [xmin, ymin, xmax, ymax]


def update_trace_indices(df: pd.DataFrame, log_name: str,
                         borders: list) -> pd.DataFrame:
    """Update trace indices to window indices for event log.

    Args:
        df (pd.DataFrame): DataFrame with drift information
        log_name (str): Name of event log
        borders (list): Window borders in form of trace indices

    Returns:
        pd.DataFrame: Updated DataFrame
    """
    df_filtered = df.loc[df["log_name"] == log_name]

    df_filtered.loc[:, "change_trace_index"] = df_filtered.loc[:, "change_trace_index"]\
        .map(lambda x: df_trace_index_2_window_id(borders, x))
    df_filtered.loc[:, "change_trace_index"] = df_filtered.loc[:, "change_trace_index"]\
        .map(get_drift_coordinates)

    df.update(df_filtered)

    return df


def df_trace_index_2_window_id(borders: list, trace_ids: list) -> list:
    """Convert trace index to window id.

    Args:
        borders (list): Window borders in form of trace indices
        trace_ids (list): Trace indices for drift

    Returns:
        list: Window ID(s) in which the drift occurred
    """
    result_set = []

    if len(trace_ids) > 1:
        drift_start = trace_ids[0]
        drift_end = trace_ids[1]

        for i, elem in enumerate(borders, 1):
            left, right = elem
            if is_in_window(left, drift_start, right):
                result_set.insert(0, i)
            elif is_in_window(left, drift_end, right):
                result_set.append(i)
            if len(result_set) == len(trace_ids):
                break
    else:
        drift_start = trace_ids[0]
        for i, elem in enumerate(borders, 1):
            left, right = elem
            if is_in_window(left, drift_start, right):
                result_set.append(i)
                break

    # trace not found in borders
    if not result_set:
        result_set.append(-100)

    return result_set


def is_in_window(left: int, x: int, right: int) -> bool:
    """Check if trace id is within window.

    Args:
        left (int): First trace ID in window
        x (int): Trace ID to check
        right (int): Last trace ID in window

    Returns:
        bool: True, if trace ID lies in window, False otherwise
    """
    return left <= x <= right


def special_string_2_list(s: str) -> list[int]:
    """Convert given string to list of integers.

    Args:
        s (str): String of format "[x,x,x,x]"

    Returns:
        list[int]: List containing integer values
    """
    return list(map(int, s.translate({ord(i): None for i in "[]"}).split(",")))


def special_string_2_list_float(s: str) -> list[float]:
    """Convert given string to list of floats.

    Args:
        s (str): String of format "[x,x,x,x]"

    Returns:
        list[float]: List containing float values
    """
    return list(map(float, s.translate({ord(i): None for i in "[]"}).split(",")))


def bbox_coco_format(box: list) -> list:
    """Convert bbox to coco format.

    Args:
        box (list): Bounding box

    Returns:
        list: Boundining box in coco format
    """
    xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]
    x = xmin
    y = ymin
    width = xmax - xmin
    height = ymax - ymin
    return [x, y, width, height]


def get_bbox_as_list_coco(df: pd.DataFrame, drift_type: str) -> list:
    """Get bbox in coco format as list.

    Args:
        df (pd.DataFrame): DataFrame containing drift information
        drift_type (str): Respective drift type

    Returns:
        list: Bounding box
    """
    if drift_type == "sudden":
        bbox = df.iloc[0]["change_trace_index"]
        bbox = get_sudden_bbox_coco(bbox)
        return bbox_coco_format(bbox)
    elif drift_type == "gradual":
        if len(df.index) > 1:
            first_row = df.iloc[0]["change_trace_index"]
            last_row = df.iloc[-1]["change_trace_index"]
        else:
            bbox = df.iloc[0]["change_trace_index"]
        bbox = get_gradual_bbox_coco(bbox)
        return bbox_coco_format(bbox)
    elif drift_type == "incremental" or drift_type == "recurring":
        first_row = df.iloc[0]["change_trace_index"]
        last_row = df.iloc[-1]["change_trace_index"]
        return bbox_coco_format(
            [first_row[0], first_row[1], last_row[2], last_row[3]])


def get_sudden_bbox_coco(bbox: list) -> list:
    """Get bbox for sudden drift in coco format. 

    Args:
        bbox (list): Bounding box of sudden drift

    Returns:
        list: Bounding box
    """
    # artificially enlarge sudden bboxes for detection
    if cfg.RESIZE_SUDDEN_BBOX and bbox[0] < cfg.RESIZE_VALUE:
        bbox[0] = 0
        bbox[1] = 0
        bbox[2] += cfg.RESIZE_VALUE
        bbox[3] += cfg.RESIZE_VALUE
    elif cfg.RESIZE_SUDDEN_BBOX and bbox[0] > cfg.RESIZE_VALUE:
        if check_window_size(bbox[3] + cfg.RESIZE_VALUE):
            bbox[0] -= cfg.RESIZE_VALUE
            bbox[1] -= cfg.RESIZE_VALUE
            bbox[2] += cfg.RESIZE_VALUE
            bbox[3] += cfg.RESIZE_VALUE
        else:
            bbox[0] -= cfg.RESIZE_VALUE
            bbox[1] -= cfg.RESIZE_VALUE
            bbox[2] = cfg.N_WINDOWS
            bbox[3] = cfg.N_WINDOWS
    else:
        # add at least 1 for width/heigh to detect drifts
        if check_window_size(bbox[3] + 1):
            bbox[2] += 1
            bbox[3] += 1
        else:
            bbox[0] - 1
            bbox[1] - 1
            bbox[2] = cfg.N_WINDOWS
            bbox[3] = cfg.N_WINDOWS
    return bbox


def get_gradual_bbox_coco(bbox: list) -> list:
    """Get bbox for gradual drift in coco format. 

    Args:
        bbox (list): Bounding box of gradual drift

    Returns:
        list: Bounding box
    """
    # add 1 if gradual drift happens in same window
    if bbox[0] == bbox[3]:
        if check_window_size(bbox[3] + 1):
            bbox[2] += 1
            bbox[3] += 1
        else:
            bbox[0] - 1
            bbox[1] - 1
            bbox[2] = cfg.N_WINDOWS
            bbox[3] = cfg.N_WINDOWS
    return bbox


def check_window_size(value: int, n_windows=cfg.N_WINDOWS) -> bool:
    """Checks if window index lies outside of number of windows.

    Args:
        value (int): Window index
        n_windows (int): Number of windows

    Returns:
        bool: True, if pixel value lies within n_windows, False otherwise
    """
    if value > n_windows:
        return False
    else:
        return True


def get_bbox_as_list_coco_untyped(df: pd.DataFrame, drift_type) -> list:
    """Get bbox in coco format as untyped list.

    Args:
        df (pd.DataFrame): DataFrame containing drift information
        drift_type (str): Respective drift type

    Returns:
        list: Bounding box
    """
    if drift_type == "sudden":
        bbox = special_string_2_list(df.iloc[0]["change_trace_index"])
        bbox = get_sudden_bbox_coco(bbox)
        return bbox_coco_format(bbox)
    elif drift_type == "gradual":
        if len(df.index) > 1:
            first_row = special_string_2_list(df.iloc[0]["change_trace_index"])
            last_row = special_string_2_list(df.iloc[-1]["change_trace_index"])
        else:
            bbox = special_string_2_list(df.iloc[0]["change_trace_index"])
        bbox = get_gradual_bbox_coco(bbox)
        return bbox_coco_format(bbox)
    elif drift_type == "incremental" or drift_type == "recurring":
        first_row = special_string_2_list(df.iloc[0]["change_trace_index"])
        last_row = special_string_2_list(df.iloc[-1]["change_trace_index"])
        return bbox_coco_format(
            [first_row[0], first_row[1], last_row[2], last_row[3]])


def get_area(width: Union[int, float], height: Union[int, float]) -> float:
    """Get area for bounding box.

    Args:
        width (Union[int, float]): Bounding box width
        height (Union[int, float]):Bounding box height

    Returns:
        float: Area
    """
    return width * height


def get_drift_id(drift_type: str) -> int:
    """Get drift ID for drift type.

    Args:
        drift_type (str): Drift type

    Returns:
        int: Drift ID for drift type
    """
    try:
        drift_id = cfg.DRIFT_TYPES.index(drift_type) + 1
    except Exception:
        f"Drift type not specified in config - drift types: {cfg.DRIFT_TYPES}"
    return drift_id


def get_segmentation(bbox: list) -> list:
    """Get segmentation values for bounding box. 
    Not used for training.

    Args:
        bbox (list): Bounding box

    Returns:
        list: Segmentation values
    """
    xmin = bbox[0]
    ymin = bbox[1]
    xmax = bbox[0] + bbox[2]
    ymax = bbox[1] + bbox[3]
    segmentation = [[xmin, ymin],
                    [xmax, ymin],
                    [xmax, ymax],
                    [xmin, ymax]]
    return list([segmentation])


def generate_annotations(drift_info: pd.DataFrame, dir: str,
                         log_matching: dict, log_names: list):
    """Generate annotations for WINSIM encoding approach. 
    Annotations are saved in JSON format.

    Args:
        drift_info (pd.DataFrame): DataFrame, containing drift information
        dir (str): Path of data directory
        log_matching (dicht): Dict, containing matching between log names and images
        log_names (list): List, containing names of all event logs
    """
    categories = [
        {"supercategory": "drift", "id": 1, "name": "sudden"},
        {"supercategory": "drift", "id": 2, "name": "gradual"},
        {"supercategory": "drift", "id": 3, "name": "incremental"},
        {"supercategory": "drift", "id": 4, "name": "recurring"}
    ]

    anno_file = {
        "categories": categories,
        "images": [],
        "annotations": []
    }

    annotation_id = 0
    img_id = 0

    for log_name in log_names:
        log_annotation = {}
        log_info = drift_info.loc[drift_info["log_name"] == log_name]
        drift_ids = pd.unique(log_info["drift_or_noise_id"])

        img_id = log_matching[log_name]
        img_name = str(img_id) + ".jpg"
        img_path = os.path.join(dir, img_name)

        img = Image.open(img_path)
        width, height = img.size
        log_img = {
            "file_name": img_name,
            "height": height,
            "width": width,
            "id": img_id
        }

        anno_file["images"].append(log_img)

        for drift_id in drift_ids:
            drift = log_info.loc[log_info["drift_or_noise_id"] == drift_id]
            drift_type = pd.unique(drift["drift_type"])[0]

            category_id = get_drift_id(drift_type)
            if cfg.ANNOTATIONS_ONLY:
                bbox = get_bbox_as_list_coco_untyped(drift, drift_type)
            else:
                bbox = get_bbox_as_list_coco(drift, drift_type)
            area = get_area(width=bbox[2], height=bbox[3])
            segmentation = get_segmentation(bbox)
            log_annotation = {
                "id": annotation_id,
                "image_id": img_id,
                "category_id": category_id,
                "label": drift_type,
                "bbox": bbox,
                "area": area,
                "iscrowd": 0,
                "ignore": 0,
                "segmentation": segmentation}
            anno_file["annotations"].append(log_annotation)
            annotation_id += 1

        img_id += 1

    annotations_path = os.path.join(dir, "annotations.json")
    with open(annotations_path, "w", encoding='utf-8') as file:
        json.dump(anno_file, file)


def read_annotations(dir: str) -> Union[list, dict]:
    """Read annotation file.

    Args:
        dir (str): Directory, where annotation file is stored.

    Returns:
        Union[list,dict]: Annotations
    """
    path = os.path.join(dir, "annotations.json")
    with open(path) as file:
        annotations = json.load(file)
    return annotations


def get_drift_info(dir: str) -> pd.DataFrame:
    """Get drift information.

    Args:
        dir (str): Directory, where drift information file is stored.

    Returns:
        pd.DataFrame: Drift information
    """
    path = os.path.join(dir, "drift_info.csv")
    return pd.read_csv(path)


def extract_drift_information(dir: str) -> pd.DataFrame:
    """Extract information from drift info file generated by CDLG.

    Args:
        dir (str): Directory of drift info file

    Returns:
        pd.DataFrame: DataFrame containing all relevant 
        information about drifts
    """
    pd_df = get_drift_info(dir)

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

    drift_info["drift_traces_index"] = drift_info["change_trace_index"]

    return drift_info


def get_ex_decoder() -> Tuple[dict, TfExampleDecoder]:
    """Get decoders for classes.

    Returns:
        Tuple[dict, TfExampleDecoder]: Category index for drift types 
        and TfExampleDecoder
    """
    category_index = {
        1: {
            "id": 1,
            "name": "sudden",
            "color": "white"
        },
        2: {
            "id": 2,
            "name": "gradual",
            "color": "dodgerblue"
        },
        3: {
            "id": 3,
            "name": "incremental",
            "color": "magenta"
        },
        4: {
            "id": 4,
            "name": "recurring",
            "color": "aqua"
        }
    }
    tf_ex_decoder = TfExampleDecoder()

    return category_index, tf_ex_decoder


def visualize_batch(path: str, mode: str, seed: int, n_examples=3):
    """Visualize bounding boxes for batch of images (groundtruth).

    Args:
        path (str): Image data path
        mode (str): Training or validation mode
        seed (int): Seed for replication
        n_examples (int, optional): Number of logs to visualize. Defaults to 3.
    """
    # dynamically create subplots based on n_examples
    columns = 3
    rows = n_examples // columns
    if n_examples % columns != 0:
        rows += 1
    pos = range(1, n_examples + 1)

    category_index, tf_ex_decoder = get_ex_decoder()

    data = tf.data.TFRecordDataset(
        path).shuffle(
        buffer_size=cfg.EVAL_EXAMPLES, seed=seed).take(n_examples)

    plt.figure(figsize=(20, 20))
    use_normalized_coordinates = True
    min_score_thresh = 0.30
    for i, serialized_example in enumerate(data):
        plt.subplot(rows, columns, pos[i])
        decoded_tensors = tf_ex_decoder.decode(serialized_example)
        image = decoded_tensors['image'].numpy().astype('uint8')
        scores = np.ones(shape=(len(decoded_tensors['groundtruth_boxes'])))
        visualization_utils.visualize_boxes_and_labels_on_image_array(
            image,
            decoded_tensors['groundtruth_boxes'].numpy(),
            decoded_tensors['groundtruth_classes'].numpy().astype(int),
            scores,
            category_index=category_index,
            use_normalized_coordinates=use_normalized_coordinates,
            max_boxes_to_draw=4,
            min_score_thresh=min_score_thresh,
            agnostic_mode=False,
            instance_masks=None,
            line_thickness=2)
        plt.imshow(image)
        plt.axis('off')
        plt.title(f'Image-{i+1}')

    plt.savefig(os.path.join(cfg.TRAINED_MODEL_PATH, f"{mode}_batch.png"),
                bbox_inches="tight")
    
    
def visualize_predictions_old(path, mode, model, seed, n_examples=3, threshold=0.50):
    """Visualize bounding boxes for batch of images (prediction).
    Is replaced by visualize_predictions.

    Args:
        path (str): Image data path
        mode (str): Training or validation mode
        model (tf.keras.Model): TensorFlow model
        seed (int): Seed for replication
        n_examples (int, optional): Number of logs to visualize. Defaults to 3.
        threshold (float, optional): Threshold for prediction confidence. 
            Defaults to 0.50.
    """
    # dynamically create subplots based on n_examples
    columns = 3
    rows = n_examples // columns
    if n_examples % columns != 0:
        rows += 1
    pos = range(1, n_examples + 1)

    input_image_size = cfg.IMAGE_SIZE
    model_fn = model.signatures['serving_default']

    category_index, tf_ex_decoder = get_ex_decoder()

    data = tf.data.TFRecordDataset(
        path).shuffle(
        buffer_size=cfg.EVAL_EXAMPLES, seed=seed).take(n_examples)

    plt.figure(figsize=(20, 20))

    for i, serialized_example in enumerate(data):
        plt.subplot(rows, columns, pos[i])
        decoded_tensors = tf_ex_decoder.decode(serialized_example)
        image = build_inputs_for_object_detection(
            decoded_tensors['image'], input_image_size)
        image = tf.expand_dims(image, axis=0)
        image = tf.cast(image, dtype=tf.uint8)
        image_np = image[0].numpy()
        result = model_fn(image)
        visualization_utils.visualize_boxes_and_labels_on_image_array(
            image_np,
            result['detection_boxes'][0].numpy(),
            result['detection_classes'][0].numpy().astype(int),
            result['detection_scores'][0].numpy(),
            category_index=category_index,
            use_normalized_coordinates=False,
            max_boxes_to_draw=200,
            min_score_thresh=threshold,
            agnostic_mode=False,
            instance_masks=None,
            line_thickness=2)
        visualization_utils.visualize_boxes_and_labels_on_image_array(
            image_np,
            decoded_tensors['groundtruth_boxes'].numpy(),
            decoded_tensors['groundtruth_classes'].numpy().astype(int),
            scores=None,
            category_index=category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=threshold,
            agnostic_mode=False,
            instance_masks=None,
            line_thickness=2)

        plt.imshow(image_np)
        plt.axis('off')

    plt.savefig(os.path.join(cfg.TRAINED_MODEL_PATH, f"{mode}_predictions.png"),
                bbox_inches="tight")


def visualize_predictions(path:str, mode: str, model: tf.keras.Model, 
                          seed: int, n_examples=3, threshold=0.50):
    """Visualize bounding boxes for batch of images (prediction).

    Args:
        path (str): Image data path
        mode (str): Training or validation mode
        model (tf.keras.Model): TensorFlow model
        seed (int): Seed for replication
        n_examples (int, optional): Number of logs to visualize. Defaults to 3.
        threshold (float, optional): Threshold for prediction confidence. 
            Defaults to 0.50.
    """
    # dynamically create subplots based on n_examples
    columns = 3
    rows = n_examples // columns
    if n_examples % columns != 0:
        rows += 1
    pos = range(1, n_examples + 1)

    input_image_size = cfg.IMAGE_SIZE
    model_fn = model.signatures['serving_default']

    category_index, tf_ex_decoder = get_ex_decoder()

    data = tf.data.TFRecordDataset(
        path).shuffle(
        buffer_size=cfg.EVAL_EXAMPLES, seed=seed).take(n_examples)

    plt.figure(figsize=(20, 20))
    for i, serialized_example in enumerate(data):
        plt.subplot(rows, columns, pos[i])
        decoded_tensors = tf_ex_decoder.decode(serialized_example)
        image = build_inputs_for_object_detection(
            decoded_tensors['image'], input_image_size)
        image = tf.expand_dims(image, axis=0)
        image = tf.cast(image, dtype=tf.uint8)
        image_np = image[0].numpy()
        result = model_fn(image)

        scores = result['detection_scores'][0].numpy()

        bbox_pred = result['detection_boxes'][0].numpy()
        bbox_pred = bbox_pred[scores > threshold]

        y_pred = result['detection_classes'][0].numpy().astype(int)
        y_pred = y_pred[scores > threshold]

        visualize_boxes_and_labels(image=image_np,
                                   bboxes=bbox_pred,
                                   labels=y_pred,
                                   score=result['detection_scores'][0].numpy(),
                                   category_index=category_index,
                                   is_groundtruth=False)
        visualize_boxes_and_labels(image=image_np,
                                   bboxes=decoded_tensors['groundtruth_boxes'].numpy(
                                   ),
                                   labels=decoded_tensors['groundtruth_classes'].numpy().astype(
                                       int),
                                   score=None,
                                   category_index=category_index,
                                   is_groundtruth=True)
        plt.imshow(image_np)
        plt.axis('off')

    plt.savefig(os.path.join(cfg.TRAINED_MODEL_PATH, f"{mode}_predictions.png"),
                bbox_inches="tight")


def visualize_boxes_and_labels(image: np.ndarray, bboxes: list, labels: list, 
                               score: list, category_index: dict, is_groundtruth: bool):
    """Visualize bounding boxes and the corresponding labels.

    Args:
        image (np.ndarray): Image as numpy array
        bboxes (list): Bounding boxes
        labels (list): Corresponding labels
        score (list): Prediction confidence values, optional
        category_index (dict): Mapping of numerical to categorical label
        is_groundtruth (bool): Indicating whether values are from 
            groundtruth or prediction
    """
    ax = plt.gca()

    for i, (box, cls) in enumerate(zip(bboxes, labels)):

        tx1, ty1, x2, y2 = box

        if is_groundtruth:
            text = "{}: GT".format(category_index[cls]["name"])
            edgecolor = "black"

            im_height, im_width, _ = image.shape

            tx1, ty1, x2, y2 = (tx1 * im_width, ty1 * im_height,
                                x2 * im_width, y2 * im_height)
            text_pos = y2 + (image.shape[0]*0.03)
        else:
            text = "{}: {}%".format(
                category_index[cls]["name"], int(np.round(score[i] * 100, 0)))
            text_pos = ty1 - (image.shape[0]*0.02)
            edgecolor = category_index[cls]["color"]

        # width (w) = xmax - xmin ; height (h) = ymax - ymin
        tw, th = x2 - tx1, y2 - ty1

        patch = plt.Rectangle(
            [tx1, ty1], tw, th, fill=False, edgecolor=edgecolor, linewidth=1
        )
        ax.add_patch(patch)
        ax.text(
            tx1,
            text_pos,
            text,
            bbox={"facecolor": edgecolor, "alpha": 0.75, "linewidth": 0},
            clip_box=ax.clipbox,
            clip_on=True,
            fontsize=8,
            # verticalalignment='top',
            color="white",
            weight='bold'
        )


def build_inputs_for_object_detection(image, input_image_size):
    """Builds Object Detection model inputs for serving.
    Source: 
    https://www.tensorflow.org/tfmodels/vision/object_detection#inference_from_trained_model
    Args:
        image (Any): Image
        input_image_size (Any): Target image size for model

    Returns:
        Any: Processed image
    """
    image, _ = resize_and_crop_image(
        image,
        input_image_size,
        padded_size=input_image_size,
        # dont scale images for visualization
        aug_scale_min=1.0,
        aug_scale_max=1.0)
    return image


def get_timestamp() -> str:
    """Get timestamp for current time.

    Returns:
        str: Timestamp as string in format "%Y%m%d-%H%M%S"
    """
    europe = pytz.timezone("Europe/Berlin")
    timestamp = dt.datetime.now(europe).strftime("%Y%m%d-%H%M%S")
    return timestamp


def get_model_config(model_dir: str) -> ExperimentConfig:
    """Get configuration for model training.

    Args:
        model_dir (str): Directory where model should be tracked

    Returns:
        ExperimentConfig: File, containing hyperparameters for model training
    """
    exp_config = exp_factory.get_exp_config(cfg.MODEL_SELECTION)

    # non adjustable for pretrained model
    IMG_SIZE = [cfg.HEIGHT, cfg.WIDTH, 3]

    # Model specific config
    if cfg.MODEL_SELECTION == "retinanet_spinenet_coco":
        exp_config.task.model.backbone.spinenet.model_id = cfg.SPINENET_ID
        exp_config.task.model.anchor.anchor_size = 4
        if cfg.SPINENET_ID == "190":
            exp_config.task.model.head.num_convs = 7
            exp_config.task.model.head.num_filters = 512

    # Backbone config
    exp_config.task.freeze_backbone = False
    exp_config.task.annotation_file = ''
    exp_config.task.init_checkpoint = ''

    # Model config
    exp_config.task.model.input_size = IMG_SIZE
    exp_config.task.model.num_classes = cfg.N_CLASSES + 1
    exp_config.task.model.detection_generator. \
        tflite_post_processing.max_classes_per_detection = \
        exp_config.task.model.num_classes

    # Training data config
    exp_config.task.train_data.input_path = cfg.TRAIN_DATA_DIR
    exp_config.task.train_data.dtype = 'float32'
    exp_config.task.train_data.global_batch_size = cfg.TRAIN_BATCH_SIZE
    exp_config.task.train_data.parser.aug_scale_max = cfg.SCALE_MAX
    exp_config.task.train_data.parser.aug_scale_min = cfg.SCALE_MIN

    # Validation data config
    exp_config.task.validation_data.input_path = cfg.EVAL_DATA_DIR
    exp_config.task.validation_data.dtype = 'float32'
    exp_config.task.validation_data.global_batch_size = cfg.EVAL_BATCH_SIZE

    train_steps = cfg.TRAIN_STEPS
    # steps_per_loop = num_of_training_examples // train_batch_size
    exp_config.trainer.steps_per_loop = cfg.STEPS_PER_LOOP

    exp_config.trainer.summary_interval = cfg.SUMMARY_INTERVAL
    exp_config.trainer.checkpoint_interval = cfg.CP_INTERVAL
    exp_config.trainer.validation_interval = cfg.VAL_INTERVAL
    # validation_steps = num_of_validation_examples // eval_batch_size
    exp_config.trainer.validation_steps = cfg.VAL_STEPS
    exp_config.trainer.train_steps = train_steps

    # Optimizer and LR config
    exp_config.trainer.optimizer_config.optimizer.type = cfg.OPTIMIZER_TYPE
    if cfg.OPTIMIZER_TYPE == "sgd":
        exp_config.trainer.optimizer_config.optimizer.sgd.clipnorm = cfg.SGD_CLIPNORM
        exp_config.trainer.optimizer_config.optimizer.sgd.momentum = cfg.SGD_MOMENTUM
    elif cfg.OPTIMIZER_TYPE == "adam":
        exp_config.trainer.optimizer_config.optimizer.adam.beta_1 = cfg.ADAM_BETA_1
        exp_config.trainer.optimizer_config.optimizer.adam.beta_2 = cfg.ADAM_BETA_2

    if cfg.LR_DECAY:
        exp_config.trainer.optimizer_config.learning_rate.type = cfg.LR_TYPE
        if cfg.LR_TYPE == "cosine":
            exp_config.trainer.optimizer_config.learning_rate.cosine.decay_steps = \
                train_steps
            exp_config.trainer.optimizer_config.learning_rate.cosine.\
                initial_learning_rate = cfg.LR_INITIAL
        elif cfg.LR_TYPE == "stepwise":
            exp_config.trainer.optimizer_config.learning_rate.stepwise.boundaries = \
                cfg.STEPWISE_BOUNDARIES
            exp_config.trainer.optimizer_config.learning_rate.stepwise.values = \
                cfg.STEPWISE_VALUES
    else:
        exp_config.trainer.optimizer_config.learning_rate.type = 'constant'
        exp_config.trainer.optimizer_config.learning_rate.constant.learning_rate = \
            cfg.LR_INITIAL

    exp_config.trainer.optimizer_config.warmup.linear.warmup_steps = cfg.LR_WARMUP_STEPS
    exp_config.trainer.optimizer_config.warmup.linear.warmup_learning_rate = \
        cfg.LR_WARMUP

    # Checkpoint strategy
    exp_config.trainer.best_checkpoint_eval_metric = cfg.BEST_CP_METRIC
    exp_config.trainer.best_checkpoint_export_subdir = os.path.join(model_dir,
                                                                    "best_cp")
    exp_config.trainer.best_checkpoint_metric_comp = cfg.BEST_CP_METRIC_COMP

    return exp_config


def matrix_to_img(matrix: np.ndarray, number: int, exp_path: str, mode="color"):
    """Convert similarity matrix to image and save it. 

    Args:
        matrix (np.array): Similarity matrx
        number (int): Image number
        exp_path (str): Experiment path
        mode (str, optional): Coloring mode. Defaults to "color".
    """
    if mode == "color":
        # Get the color map by name:
        cm = plt.get_cmap('viridis')
        # Apply the colormap like a function to any array:
        colored_image = cm(matrix)

        im = Image.fromarray((colored_image[:, :, :3] * 255).astype(np.uint8))

    elif mode == "gray":
        im = Image.fromarray(matrix).convert("RGB")

    im.save(os.path.join(exp_path, f"{number}.jpg"))


def start_tfr_script(repo_dir: str, data_dir: str, tfr_dir: str, prefix: str):
    """!! BE CAREFUL, THIS PART IS HARDCODED !!
    To use this function, you must clone the Tensorflow model garden clone repository, 
    which can be found here: https://github.com/tensorflow/models/tree/master
    
    Args:
        repo_dir (str): Path of the Tensorflow model garden repository
        data_dir (str): Path of image directory
        tfr_dir (str): Path of tfr directory
        prefix (str): Prefix for naming tfr file
    """

    annotations_path = os.path.join(data_dir, "annotations.json")
    output_path = os.path.join(tfr_dir, prefix)

    cmd = f"python -m official.vision.data.create_coco_tf_record --logtostderr \
            --image_dir={data_dir} \
            --object_annotations_file={annotations_path} \
            --output_file_prefix={output_path} \
            --num_shards=1"
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True, cwd=repo_dir)

    try:
        outs, errs = p.communicate(timeout=15)
    except subprocess.TimeoutExpired:
        p.kill()
        outs, errs = p.communicate()


def load_image(path: str) -> np.ndarray:
    """Load image into numpy array.

    Args:
        path (str): Image path

    Returns:
        np.ndarray: Image as numpy array
    """
    image = Image.open(path)
    return np.array(image)


def datetime_2_str(date: dt.datetime) -> str:
    """Convert datetime object to string.

    Args:
        date (dt.datetime): Datetime object

    Returns:
        str: String representation of datime object in format "%m-%d-%Y"
    """
    return dt.datetime.strftime(date, "%m-%d-%Y")


def get_all_numbers_of_traces_per_log():
    """Get all numbers of traces per event log.
    """
    files = get_event_log_paths()
    number_per_log = {}
    for name, path in tqdm(files.items(), desc="Counting Traces from Event Logs",
                           unit="Event Log"):
        log = pm.read_xes(os.path.join(path, name),
                          return_legacy_log_object=True)
        number_per_log[name] = len(log)
    number_of_traces_path = os.path.join(
        cfg.TEST_IMAGE_DATA_DIR, "number_of_traces.json")
    with open(number_of_traces_path, "w", encoding='utf-8') as file:
        json.dump(number_per_log, file)


def get_number_of_traces(event_log: EventLog) -> int:
    """Get number of traces for event log.

    Args:
        event_log (EventLog): Event log

    Returns:
        int: Number of traces
    """
    return len(event_log)


def create_winsim_experiment(dir: str) -> str:
    """Create experiment folder structure for WINSIM encoding.

    Args:
        dir (str): Path where experiment should be stored

    Returns:
        path: Experiment path
    """
    timestamp = get_timestamp()

    exp_path = os.path.join(dir, "winsim", f"experiment_{timestamp}")
    os.makedirs(exp_path)

    print(f"Experiment created at {exp_path}")
    return exp_path


def check_dfg_graph_freq(log:pd.DataFrame) -> float:
    graph, sa, ea = discover_dfg_typed(log)
    
    freq = sum(graph.values())
    
    return freq