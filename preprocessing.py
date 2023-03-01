import pm4py as pm
import numpy as np
import pandas as pd
from pm4py import discover_dfg_typed
from numpy import linalg as LA
from scipy import spatial


def preprocessing(event_log_path: str, n_windows: int):
    event_log = pm.read_xes(event_log_path)

    windowed_dfg_matrices = log_to_windowed_dfg(event_log, n_windows)

    sim_matrices = similarity_calculation(windowed_dfg_matrices)

    return sim_matrices


def log_to_windowed_dfg(event_log, n_windows):

    act_names = np.unique(event_log["concept:name"])

    window_size = len(event_log) // n_windows + 1
    dfg_graphs = []
    # start_act = []
    # end_act = []
    left_boundary = 0
    right_boundary = window_size
    for _ in range(1, n_windows + 1):
        dfg_matrix_df = pd.DataFrame(0, columns=act_names, index=act_names)

        log_window = event_log[left_boundary:right_boundary]

        graph, sa, ea = discover_dfg_typed(log_window)

        for relation, freq in graph.items():
            rel_a, rel_b = relation
            dfg_matrix_df.at[rel_a, rel_b] = freq

        dfg_graphs.append(dfg_matrix_df)
        # start_act.append(sa)
        # end_act.append(ea)

        left_boundary = right_boundary
        right_boundary += window_size

    return np.array(dfg_graphs)  # , np.array(start_act), np.array(end_act)


def similarity_calculation(windowed_dfg):

    # create matrix
    n = len(windowed_dfg)
    sim_matrix = np.zeros((n, n))

    for i, matrix_i in enumerate(windowed_dfg):
        for j, matrix_j in enumerate(windowed_dfg):
            sim_matrix[i, j] = calc_distance_norm(matrix_i, matrix_j)

    # check diagonal of similarity matrix
    assert np.sum(np.diagonal(sim_matrix)
                  ) == 0, "The diagonal of the similarity matrix must be zero."

    norm_sim_matrix = 1 - sim_matrix / np.amax(sim_matrix)

    return norm_sim_matrix


def calc_distance_norm(matrix_1, matrix_2, option="l2"):
    diff = matrix_1 - matrix_2
    if option == "fro":
        # Frobenius norm
        dist_value = LA.norm(diff, "fro")
    elif option == "nuc":
        # nuclear norm
        dist_value = LA.norm(diff, "nuc")
    elif option == "inf":
        # max norm
        dist_value = LA.norm(diff, "inf")
    elif option == "l2":
        # L2 norm
        dist_value = LA.norm(diff, 2)
    elif option == "cosine":
        # cosine distance
        dist_value = spatial.distance.cosine(
            matrix_1.ravel(), matrix_2.ravel())

    if np.isnan(dist_value):
        dist_value = LA.norm(diff, "fro")

    return dist_value
