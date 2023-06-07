import os
import csv
import copy
import math
import subprocess
import pandas as pd
import seaborn as sns
import object_segmentation.utils.config as cfg

from pathlib import Path


def vdd_draw_drift_map_with_clusters(data, number, exp_path, 
                                     cmap="plasma", y_lines=None):
    ''' the main script that describes drawing of the DriftMAP
    Source:
    https://github.com/yesanton/Process-Drift-Visualization-With-Declare/blob/master/src/visualize_drift_map.py
    Author: Anton Yeshchenko
    '''
    data_c = copy.deepcopy(data)
    for i in range(len(data)):
        for j in range(len(data[i])):
            data_c[i][j] = data[i][j]/100

    # https://matplotlib.org/examples/color/colormaps_reference.html
    # copper
    # viridis
    ax = sns.heatmap(data_c,
                     linewidth=0,
                     cmap=cmap,
                     xticklabels=False,
                     yticklabels=False,
                     cbar=False)

    # in case of debug to show directly here
    # plt.show()

    # draw horizontal lines
    lines = [y_lines] * (len(data_c[0])+1)
    dataT = pd.DataFrame(lines)
    _ = sns.lineplot(data=dataT, legend=False, palette=['white'] * len(
        y_lines), dashes=[(2, 2)] * len(y_lines), linewidth=1)

    # # here is the same but faster
    # if x_lines_all:
    #     if len(x_lines_all) == 1:
    #         ax.vlines(next(iter(x_lines_all.values())), *ax.get_ylim(),
    #                   colors='white', linestyles='-.', linewidth=1)
    #     # here draw per cluster
    #     else:
    #         to_ind = 0

    #         for i, j in zip(cluster_order, y_lines):
    #             from_ind = to_ind
    #             to_ind = j
    #             for k in x_lines_all[i]:
    #                 plt.plot([k, k], [from_ind, to_ind],
    #                          linestyle='-.', color='white', linewidth=1)
   # ax.tight_layout()
    ax.get_figure().savefig(os.path.join(exp_path, f"{number}.jpg"),
                            bbox_inches='tight')


def vdd_mine_minerful_for_declare_constraints(log_name:str, path, exp_path):
    '''
    Source: 
    https://github.com/yesanton/Process-Drift-Visualization-With-Declare/blob/master/src/auxiliary/minerful_adapter.py
    Author: Anton Yeshchenko
    '''
    env = dict(os.environ)
    env['JAVA_OPTS'] = 'foo'
    subprocess.call(['java', '-version'])

    # changed saving and data structure strategy
    cwd = os.path.join(os.getcwd(), "approaches", "vdd", "src", "minerful_scripts")
    
    naming = log_name.split(".")[0]
    
    file_input = os.path.abspath(path)
    
    csv_dir = os.path.abspath(os.path.join(exp_path,"minerful_constraints"))
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
                     #we always start from the first
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
    if constraint_type not in set(['confidence', 'support', \
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
    '''
    # every first timestamp of each trace is stored here
    try:
        timestamps = [trace._list[0]._dict['time:timestamp'].strftime(
            '%m-%d-%Y') for trace in log._list]
    except AttributeError:
        timestamps = [trace._list[0]._dict['time:timestamp'][0:8]
                      for trace in log._list]
    time_out = []
    n_th = 0
    number_of_timestamps = (len(timestamps) - cfg.SUB_L) / cfg.SLI_BY
    skip_every_n_th = math.ceil(number_of_timestamps / 30)

    # timestamps = sorted(timestamps)
    for i in range(0, len(timestamps) - cfg.SUB_L, cfg.SLI_BY):
        # print (timestamps[i] + " for from: " + str(i) + ' to ' + str(i+window_size))
        # results_writer.writerow([timestamps[i]])
        if n_th % skip_every_n_th == 0:
            time_out.append(timestamps[i])
        else:
            time_out.append(" ")
        n_th += 1
    return time_out