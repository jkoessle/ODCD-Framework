import os
import datetime
import pytz
import utils.config as cfg
import pm4py as pm
import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.log.obj import EventLog
from pm4py.algo.filtering.log.attributes import attributes_filter
from deprecated import deprecated
from pathlib import Path
from sklearn.preprocessing import MultiLabelBinarizer


def get_event_log_paths():
    list_of_files = {}
    for dir_path, dir_names, filenames in os.walk(cfg.DEFAULT_LOG_DIR):
        for filename in filenames:
            if filename.endswith('.xes'):
                list_of_files[filename] = dir_path

    assert len(list_of_files) > 0, f"{cfg.DEFAULT_LOG_DIR} is empty"

    return list_of_files


def import_event_log(path, name):
    variant = xes_importer.Variants.ITERPARSE
    parameters = {variant.value.Parameters.TIMESTAMP_SORT: True,
                  variant.value.Parameters.SHOW_PROGRESS_BAR: False}
    event_log = xes_importer.apply(os.path.join(
        path, name), variant=variant, parameters=parameters)

    return event_log


def read_event_log(path, name):
    event_log = pm.read_xes(os.path.join(
        path, name))

    return pl.DataFrame(event_log)


def filter_complete_events(log: EventLog):
    try:
        filtered_log = attributes_filter.apply_events(log, ["COMPLETE"], parameters={
            attributes_filter.Parameters.ATTRIBUTE_KEY: "lifecycle:transition",
            attributes_filter.Parameters.POSITIVE: True})
    except Exception:
        filtered_log = log

    return filtered_log


def export_nested_log_information(log_info):
    pass


def get_nested_log_information(log: EventLog) -> tuple[dict, dict]:
    """Get metadata information of given event log generated by CDLG tool. 
    Returns information on drift and noise generated during log creation.

    Args:
        log (EventLog): EventLog generated with CDLG

    Returns:
        tuple[dict, dict]: two dicts containing information on drift and noise
    """

    #TODO: workaround - CDLG currently only supports noise info for logs without drift
    try:
        drift_info = log.attributes["drift:info"]["children"]
        noise_info = log.attributes["noise:info"]["children"]
    except Exception:
        drift_info = {"drift_type": "no-drift"}
        noise_info = log.attributes["noise:info"]["children"]

    return noise_info, drift_info


@deprecated(version='0.1', reason="This function was for a previous version of CDLG")
def get_collection_information() -> pl.DataFrame:
    """Return polars DataFrame with information about event log

    Returns:
        pl.DataFrame: contains metadata of event log
    """
    path = os.path.join(cfg.DEFAULT_LOG_DIR, "collection_info.csv")

    return pl.read_csv(path)


def matrix_to_img(matrix, number, drift_type, exp_path, mode="color"):

    if mode == "color":
        # Get the color map by name:
        cm = plt.get_cmap('viridis')
        # Apply the colormap like a function to any array:
        colored_image = cm(matrix)

        im = Image.fromarray((colored_image[:, :, :3] * 255).astype(np.uint8))

    elif mode == "gray":
        im = Image.fromarray(matrix).convert("RGB")

    if cfg.MULTILABEL:
        im.save(os.path.join(exp_path, f"{number}_{drift_type}.png"))
    else:
        # save image with specified drift type
        if drift_type == "gradual":
            im.save(os.path.join(exp_path,
                    "gradual", f"gradual_{number}.png"))
        elif drift_type == "sudden":
            im.save(os.path.join(exp_path,
                    "sudden", f"sudden_{number}.png"))
        elif drift_type == "incremental":
            im.save(os.path.join(exp_path,
                    "incremental", f"incremental_{number}.png"))
        elif drift_type == "recurring":
            im.save(os.path.join(exp_path,
                    "recurring", f"recurring_{number}.png"))
        elif drift_type == "no_drift":
            im.save(os.path.join(exp_path,
                    "no_drift", f"no_drift_{number}.png"))
        elif drift_type == "eval":
            im.save(os.path.join(exp_path,
                    "eval", f"eval_{number}.png"))


def get_timestamp():
    europe = pytz.timezone("Europe/Berlin")
    timestamp = datetime.datetime.now(europe).strftime("%Y%m%d-%H%M%S")
    return timestamp


def create_experiment():

    timestamp = get_timestamp()

    exp_path = os.path.join(cfg.DEFAULT_DATA_DIR, f"experiment_{timestamp}")
    cwd = os.getcwd()

    for drift in cfg.DRIFT_TYPES:
        path = os.path.join(cwd, exp_path, drift)
        os.makedirs(path)

    print(f"Experiment created at {exp_path}")
    return exp_path


def create_multilabel_experiment():

    timestamp = get_timestamp()

    exp_path = os.path.join(cfg.DEFAULT_DATA_DIR, f"experiment_{timestamp}")
    os.makedirs(exp_path)

    print(f"Experiment created at {exp_path}")
    return exp_path


def create_output_directory(timestamp):

    cwd = os.getcwd()
    out_path = os.path.join(cwd,
                            cfg.DEFAULT_OUTPUT_DIR,
                            f"{timestamp}_{cfg.MODEL_SELECTION}")

    os.makedirs(out_path)
    os.makedirs(os.path.join(out_path, "images"))

    return out_path


def generate_multilabel_info(dir):
    list_of_files = [f for f in os.listdir(dir) if f.endswith(".png")]

    multilabels = []

    # get labels of images based on filename
    for file in list_of_files:
        filename = Path(file).stem
        labels = filename.split("_")[1:]
        labels_idx = [cfg.DRIFT_TYPES.index(elem) for elem in labels]
        multilabels.append(labels_idx)

        # multilabels.append(tuple(labels))

    mlb = MultiLabelBinarizer(sparse_output=False)
    one_hot_enc = mlb.fit_transform(multilabels)

    # create label lookup as csv
    labels = pd.DataFrame(one_hot_enc, columns=cfg.DRIFT_TYPES)
    labels.insert(loc=0, column="filenames", value=list_of_files)
    labels.to_csv(os.path.join(dir, "labels.csv"), index=False, sep=",")


def onehot_2_string_labels(label, label_categorical):
    labels = []
    for i, label in enumerate(label):
        if label_categorical[i]:
            labels.append(label)
    if len(labels) == 0:
        labels.append("NONE")
    return labels


def show_samples(dataset):
    fig = plt.figure(figsize=(10, 10))
    # take the first batch of dataset
    for img, label in dataset.take(1):
        # show images of first batch
        for i in range(cfg.BATCH_SIZE):
            _ = plt.subplot(6, 6, i + 1)
            plt.imshow(img[i].numpy().astype("uint8"))
            plt.title("(" + str(label[i].numpy()) + ") " +
                      str(onehot_2_string_labels(cfg.DRIFT_TYPES, label[i].numpy())))
            plt.axis("off")
    fig.tight_layout()
    plt.show()
