import os
import json
import pandas as pd
from . import config as cfg
from PIL import Image


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

    # return np.array((xmin, ymin, xmax, ymax))
    return [xmin, ymin, xmax, ymax]


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
    return [x, y, width, height]


def bbox_center_to_corner(box):
    x, y, width, height = box[0], box[1], box[2], box[3]
    xmin = x - 0.5 * width
    ymin = y - 0.5 * height
    xmax = x + 0.5 * width
    ymax = y + 0.5 * height
    return [xmin, ymin, xmax, ymax]


def get_bbox_as_list(df: pd.DataFrame, annotation_type="coco"):

    if len(df.index) > 1:
        first_row = df.iloc[0]["change_trace_index"]
        last_row = df.iloc[-1]["change_trace_index"]

        if annotation_type == "coco":
            return bbox_corner_to_center(
                [first_row[0], first_row[1], last_row[2], last_row[3]])
        else:
            return [first_row[0], first_row[1], last_row[2], last_row[3]]
    else:
        if annotation_type == "coco":
            return bbox_corner_to_center(df.iloc[0]["change_trace_index"])
        else:
            return df.iloc[0]["change_trace_index"]
        

def get_bbox_as_list_untyped(df: pd.DataFrame, annotation_type="coco"):

    if len(df.index) > 1:
        first_row = special_string_2_list(df.iloc[0]["change_trace_index"])
        last_row = special_string_2_list(df.iloc[-1]["change_trace_index"])

        if annotation_type == "coco":
            return bbox_corner_to_center(
                [first_row[0], first_row[1], last_row[2], last_row[3]])
        else:
            return [first_row[0], first_row[1], last_row[2], last_row[3]]
    else:
        if annotation_type == "coco":
            return bbox_corner_to_center(special_string_2_list(
                df.iloc[0]["change_trace_index"]))
        else:
            return special_string_2_list(df.iloc[0]["change_trace_index"])


def get_area(width, height):
    return width * height


def get_drift_id(drift_type):
    try:
        drift_id = cfg.DRIFT_TYPES.index(drift_type)
    except Exception:
        f"Drift type not specified in config - drift types: {cfg.DRIFT_TYPES}"
    return drift_id


def get_segmentation(bbox):
    bbox = bbox_center_to_corner(bbox)
    xmin = bbox[0]
    ymin = bbox[1]
    xmax = bbox[2]
    ymax = bbox[3]
    segmentation = [[xmin, ymin],
                    [xmax, ymin],
                    [xmax, ymax],
                    [xmin, ymax]]
    return list([segmentation])

def generate_annotations(drift_info, dir, log_matching):

    log_names = pd.unique(drift_info["log_name"])

    categories = [
        {"supercategory": "drift", "id": 0, "name": "sudden"},
        {"supercategory": "drift", "id": 1, "name": "gradual"},
        {"supercategory": "drift", "id": 2, "name": "incremental"},
        {"supercategory": "drift", "id": 3, "name": "recurring"}
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
            bbox = get_bbox_as_list(drift)
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

        
def read_annotations(dir):
    path = os.path.join(dir, "annotations.json")
    with open(path) as file:
        annotations = json.load(file)
    return annotations


def get_drift_info(dir) -> pd.DataFrame:
    path = os.path.join(dir, "drift_info.csv")
    return pd.read_csv(path)


def extract_drift_information(dir) -> pd.DataFrame:

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

    return drift_info