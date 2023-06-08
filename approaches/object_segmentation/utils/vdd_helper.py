import os
import csv
import copy
import math
import subprocess
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.dates as mdates
import object_segmentation.utils.config as cfg
import object_segmentation.utils.utilities as utils

from pathlib import Path
from matplotlib import pyplot as plt
from datetime import datetime


def vdd_draw_drift_map_with_clusters(data, number, exp_path, ts_ticks,
                                     timestamps, cmap="plasma"):
    ''' the main script that describes drawing of the DriftMAP
    Source:
    https://github.com/yesanton/Process-Drift-Visualization-With-Declare/blob/master/src/visualize_drift_map.py
    Author: Anton Yeshchenko
    Note: Adapted for the use of this work.
    '''

    data_c = copy.deepcopy(data)
    for i in range(len(data)):
        for j in range(len(data[i])):
            data_c[i][j] = data[i][j]/100

    ts_ticks_date = np.array(
        [datetime.strptime(d, '%m-%d-%Y').date() for d in ts_ticks])

    min_date = np.min(ts_ticks_date)
    max_date = np.max(ts_ticks_date)

    y_data = np.array(data_c)

    ax = plt.gca()
    ax.set_axis_off()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d-%Y'))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.set_xticks(ts_ticks_date)

    ax.imshow(y_data, cmap=cmap, interpolation='nearest',
              extent=[min_date, max_date, y_data.shape[0], 0], aspect='auto')

    width, height = plt.gcf().get_size_inches()
    size = (width, height)

    bboxes = {}

    for key, value in timestamps.items():
        start, end = value
        start = datetime.strftime(pd.to_datetime(start), "%m-%d-%Y")
        end = datetime.strftime(pd.to_datetime(end), "%m-%d-%Y")

        start = datetime.strptime(start, '%m-%d-%Y').date()
        end = datetime.strptime(end, '%m-%d-%Y').date()

        if start == end:
            # commented out lines are for debugging
            # line = ax.axvline(start, color='red', alpha=1.0)
            # start_date = line.get_xdata()[0]
            bboxes[key] = get_bbox_coordinates(start,
                                               start,
                                               min_date,
                                               max_date,
                                               size)
        else:
            # commented out lines are for debugging
            # line_s = ax.axvline(start, color='red', alpha=1.0, linewidth=1)
            # line_e = ax.axvline(end, color='red', alpha=1.0, linewidth=1)
            # start_date = line_s.get_xdata()[0]
            # end_date = line_e.get_xdata()[0]
            bboxes[key] = get_bbox_coordinates(start,
                                               end,
                                               min_date,
                                               max_date,
                                               size)
    plt.savefig(os.path.join(exp_path, f"{number}.jpg"),
                bbox_inches='tight',
                pad_inches=0
                )

    return bboxes


def vdd_mine_minerful_for_declare_constraints(log_name: str, path, exp_path):
    '''
    Source: 
    https://github.com/yesanton/Process-Drift-Visualization-With-Declare/blob/master/src/auxiliary/minerful_adapter.py
    Author: Anton Yeshchenko
    Note: Adapted for the use of this work.
    '''
    env = dict(os.environ)
    env['JAVA_OPTS'] = 'foo'
    subprocess.call(['java', '-version'])

    # changed saving and data structure strategy
    cwd = os.path.join(os.getcwd(), "approaches",
                       "vdd", "src", "minerful_scripts")

    naming = log_name.split(".")[0]

    file_input = os.path.abspath(path)

    csv_dir = os.path.abspath(os.path.join(exp_path, "minerful_constraints"))
    Path(csv_dir).mkdir(parents=True, exist_ok=True)

    file_output = os.path.join(csv_dir, f"{naming}.csv")

    window_size = str(cfg.SUB_L)
    sliding_window_size = str(cfg.SLI_BY)

    subprocess.call(['java',
                     "-Xmx16G",
                     "--add-modules",
                     "java.xml.bind",
                     '-cp',
                     'MINERful.jar',
                     'minerful.MinerFulMinerSlider',
                     "-iLF",
                     file_input,
                     "-iLStartAt",
                     # 0 here is at which timestamp we start,
                     # we always start from the first
                     "0",
                     "-iLSubLen",
                     window_size,
                     "-sliBy",
                     sliding_window_size,
                     '-para',
                     '4',
                     '-s',
                     '0.000000001',
                     '-c',
                     '0.0',
                     '-i',
                     '0.0',
                     '-prune',
                     # this is the pruning or not pruning options of constraints
                     'none',
                     '-sliOut',
                     file_output],
                    env=env,
                    cwd=cwd)
    return file_output


def vdd_import_minerful_constraints_timeseries_data(path, constraint_type="confidence"):
    ''' 
    Source: 
    https://github.com/yesanton/Process-Drift-Visualization-With-Declare/blob/master/src/data_importers/import_csv.py
    Author: Anton Yeshchenko
    '''
    csvfile = open(path, 'r')
    csv_reader = csv.reader(csvfile, delimiter=';', quotechar='|')

    hea = next(csv_reader, None)
    hea2 = next(csv_reader, None)

    hea2 = hea2[2:]
    hea = hea[2:]

    header_output = list()

    for i in range(len(hea)):
        if i % 3 == 0:
            tem_h = [hea2[i][1:-1]]
            temp = hea[i]
            if temp[0] == '\'':
                temp = temp[1:]
            if temp[-1] == '\'':
                temp = temp[:-1]
            if temp[-1] == ')':
                temp = temp[:-1]
            # now we split the string
            name_of_constraint_end_index = temp.find('(')
            tem_h.append(temp[:name_of_constraint_end_index])
            temp = temp[name_of_constraint_end_index+1:]
            # find if we have two events or one
            separated_constraints_index = temp.find(', ')
            if not separated_constraints_index == -1:
                tem_h.append(temp[:separated_constraints_index])
                tem_h.append(temp[separated_constraints_index+1:])
            else:
                tem_h.append(temp)
                tem_h.append('')
        else:
            tem_h = [hea2[i][1:-1]] + tem_h[1:]

        header_output.append(tem_h)

    sequences = list()

    # -2 is for the first two columns
    for i in range(len(hea)):
        sequences.append(list())

    corresponding_number_of_traces = []
    n_lines = 0
    for r in csv_reader:
        corresponding_number_of_traces.append(r[:2])
        n_lines += 1
        counter = 0
        for i in range(len(r)):
            if counter > 1:
                sequences[i-2].append(float(r[i]))
            else:
                counter += 1

    # For now we only concentrate on confidence as it is most representative measure
    if constraint_type not in set(['confidence', 'support',
                                   'interestFactor']):
        raise ValueError(constraint_type +
                         " is not a constraint type")
    # elif algoPrmts.constraint_type_used == 'confidence':
    #     cn = "Confidence"
    # elif algoPrmts.constraint_type_used == 'support':
    #     cn = "Support"
    # else:
    #     cn = "InterestF"

    constraints = []
    for i, j in zip(sequences, header_output):
        if j[0] == "Confidence":
            constraints.append(j[1:] + i)

    return constraints


def vdd_save_separately_timestamp_for_each_constraint_window(log):
    ''' 
    Source: 
    https://github.com/yesanton/Process-Drift-Visualization-With-Declare/blob/master/src/auxiliary/mine_features_from_data.py
    Author: Anton Yeshchenko
    Note: Adapted for the use of this work.
    '''
    # every first timestamp of each trace is stored here
    try:
        timestamps = [trace._list[0]._dict['time:timestamp'].strftime(
            '%m-%d-%Y') for trace in log._list]
    except AttributeError:
        timestamps = [trace._list[0]._dict['time:timestamp'][0:8]
                      for trace in log._list]
    time_out = []
    # n_th = 0
    # number_of_timestamps = (len(timestamps) - cfg.SUB_L) / cfg.SLI_BY
    # skip_every_n_th = math.ceil(number_of_timestamps / 30)

    # timestamps = sorted(timestamps)
    for i in range(0, len(timestamps) - cfg.SUB_L, cfg.SLI_BY):
        # print (timestamps[i] + " for from: " + str(i) + ' to ' + str(i+window_size))
        # results_writer.writerow([timestamps[i]])
        # if n_th % skip_every_n_th == 0:
        #     time_out.append(timestamps[i])
        # else:
        #     time_out.append(" ")
        # n_th += 1
        time_out.append(timestamps[i])
    return time_out


def extract_vdd_drift_information(dir) -> pd.DataFrame:

    pd_df = utils.get_drift_info(dir)

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

    timestamps_start = pd_df[pd_df["drift_sub_attribute"] == "change_start"]
    timestamps_start = timestamps_start[[
        "log_name", "drift_or_noise_id", "drift_attribute", "value"]]
    timestamps_start = timestamps_start.rename(
        columns={"value": "change_start"})

    timestamps_end = pd_df[pd_df["drift_sub_attribute"] == "change_end"]
    timestamps_end = timestamps_end[[
        "log_name", "drift_or_noise_id", "drift_attribute", "value"]]
    timestamps_end = timestamps_end.rename(columns={"value": "change_end"})

    drift_info = drift_info.merge((timestamps_start.merge(timestamps_end, on=[
                                  "log_name", "drift_or_noise_id", "drift_attribute"])),
                                  on=["log_name", "drift_or_noise_id",
                                      "drift_attribute"])

    drift_info["change_trace_index"] = drift_info["change_trace_index"].map(
        utils.special_string_2_list)

    drift_info["drift_traces_index"] = drift_info["change_trace_index"]

    return drift_info


def get_drift_moments_timestamps(log_name: str, drift_info: pd.DataFrame) -> dict:
    log_info = drift_info.loc[drift_info["log_name"] == log_name]
    drift_ids = pd.unique(log_info["drift_or_noise_id"])
    timestamps = {}
    for drift_id in drift_ids:
        drift = log_info.loc[log_info["drift_or_noise_id"] == drift_id]
        if len(drift.index) == 1:
            timestamp_start = drift.iloc[0]["change_start"]
            timestamp_end = drift.iloc[0]["change_end"]
            timestamps[drift_id] = (timestamp_start, timestamp_end)
        else:
            timestamp_start = drift.iloc[0]["change_start"]
            timestamp_end = drift.iloc[-1]["change_end"]
            timestamps[drift_id] = (timestamp_start, timestamp_end)

    return timestamps


def get_bbox_coordinates(start: datetime, end: datetime,
                         min: datetime, max: datetime,
                         size: tuple) -> list:
    relative_start = (mdates.date2num(start) - mdates.date2num(min)) / \
        (mdates.date2num(max) - mdates.date2num(min))
    relative_end = (mdates.date2num(end) - mdates.date2num(min)) / \
        (mdates.date2num(max) - mdates.date2num(min))

    width, height = size
    start = int(relative_start * width)
    end = int(relative_end * width)

    # ymin = 0, ymax = 1
    # format [xmin, ymin, xmax, ymax]
    bbox = [start, 0, end, height]
    return bbox


def update_drift_info_for_vdd(drift_info):
    #TODO write bbox in drift info for annotations
    pass