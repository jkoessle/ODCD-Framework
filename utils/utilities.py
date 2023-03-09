import os
import utils.config as config
import pm4py as pm
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pm4py.objects.log.importer.xes import importer as xes_importer


def get_event_log_paths():
    list_of_files = {}
    for dir_path, dir_names, filenames in os.walk(config.DEFAULT_LOG_DIR):
        for filename in filenames:
            if filename.endswith('.xes'):
                list_of_files[filename] = dir_path

    assert len(list_of_files) > 0, f"{config.DEFAULT_LOG_DIR} is empty"

    return list_of_files


def import_event_log(path, name):
    variant = xes_importer.Variants.ITERPARSE
    parameters = {variant.value.Parameters.TIMESTAMP_SORT: True}
    event_log = xes_importer.apply(os.path.join(
        path, name), variant=variant, parameters=parameters)

    return event_log


def read_event_log(path, name):
    event_log = pm.read_xes(os.path.join(
        path, name))

    return pl.DataFrame(event_log)


def get_collection_information():
    path = os.path.join(config.DEFAULT_LOG_DIR, "collection_info.csv")

    return pl.read_csv(path)


def matrix_to_img(matrix, number, drift_type, mode="color"):
    
    if mode == "color":
        # Get the color map by name:
        cm = plt.get_cmap('viridis')
        # Apply the colormap like a function to any array:
        colored_image = cm(matrix)
        
        im = Image.fromarray((colored_image[:, :, :3] * 255).astype(np.uint8))
    
    elif mode == "gray":
        im = Image.fromarray(matrix).convert("RGB")

    if drift_type == "gradual":
        im.save(os.path.join(config.DEFAULT_DATA_DIR,
                "gradual", f"gradual_{number}.png"))
    elif drift_type == "sudden":
        im.save(os.path.join(config.DEFAULT_DATA_DIR,
                "sudden", f"sudden_{number}.png"))
    elif drift_type == "none":
        im.save(os.path.join(config.DEFAULT_DATA_DIR,
                "no_drift", f"no_drift_{number}.png"))
