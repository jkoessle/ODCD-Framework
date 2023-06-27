"""The implementation and especially the assignment of true and predicted changepoints 
for evaluation is based on CDrift.
Source: https://github.com/cpitsch/cdrift-evaluation
Author: cpitsch
"""

from typing import List, Tuple
from pulp import LpProblem, LpMinimize, LpMaximize, LpVariable, LpBinary, \
    lpSum, PULP_CBC_CMD


def getTP_FP(detected: List[int], known: List[int], lag: int,
             count_duplicate_detections: bool = True) -> Tuple[int, int]:
    """Source: 
    https://github.com/cpitsch/cdrift-evaluation/blob/main/cdrift/evaluation.py
    Author: cpitsch
    Returns the number of true and false positives, using assign_changepoints to 
    calculate the assignments of detected change point to actual change point.

    Args:
        detected (List[int]): List of indices of detected change point locations.
        known (List[int]): The ground truth; List of indices of actual change points.
        lag (int): The maximal distance a detected change point can have to an actual 
        change point, whilst still counting as a true positive.
        count_duplicate_detections (bool, optional): If a detected change point is not 
        assigned to a ground truth change point, but lies within the lag window of some 
        ground truth change point, should it be counted as a false positive 
        (True if yes, False if no). Defaults to True.
    Returns:
        Tuple[int,int]: Tuple of: (true positives, false positives)

    Examples:
    >>> getTP_FP([1000,1001,2000], [1000,2000], 200, True)
    >>> (2,1)

    >>> getTP_FP([1000,1001,2000], [1000,2000], 200, False)
    >>> (2,0)
    """
    assignments = assign_changepoints(detected, known, lag_window=lag)

    # Every assignment is a True Positive,
    # and every detected point is assigned at most once
    TP = len(assignments)
    if count_duplicate_detections:
        FP = len(detected) - TP
    else:
        true_positive_candidates = [d for d in detected if any(
            (k-lag <= d and d <= k+lag) for k in known)]
        FP = len(detected) - len(true_positive_candidates)
    return (TP, FP), assignments


def assign_changepoints(detected_changepoints: List[Tuple[int, int]], 
                        actual_changepoints: List[Tuple[int, int]], 
                        lag_window: int = 200) -> List[Tuple[int, int]]:
    """Source: 
    https://github.com/cpitsch/cdrift-evaluation/blob/main/cdrift/evaluation.py
    Author: cpitsch
    Note: The code was adapted so that it also works with two-dimensional tuples or 
    change points. Therefore, the description of the function has also been changed.
    Assigns detected changepoints to actual changepoints using a LP.
    With restrictions: 

    - Detected start and end point must be within `lag_window` of actual point. 
    - Detected tuple can only be assigned to one actual tuple.
    - Every actual tuple can have at most one detected tuple assigned. 

        This is done by first optimizing for the number of assignments, finding how 
        many detected change points could be assigned, without minimizing the 
        total lag. Then, the LP is solved again, minimizing the sum of squared lags, 
        while keeping the number of assignments as high as possible.

    Args:
        detected_changepoints (List[Tuple[int, int]]): List of locations of detected 
            changepoints.
        actual_changepoints (List[Tuple[int, int]]): List of locations of actual 
            changepoints.
        lag_window (int, optional): How close must a detected change point be to an 
        actual changepoint to be a true positive. Defaults to 200.

    Examples:
    >>> detected_changepoints = [(1050, 1060), (934,934), (2100,2500)]
    >>> actual_changepoints = [(1000,1120),(1149,1149),(2000,2400)]
    >>> assign_changepoints(detected_changepoints, actual_changepoints, lag_window=200)
    >>> [((1050,1060),(1149,1149)),((934,934),(1000,1120)),((2100,2500),(2000,2400))]

    Returns:
        List[Tuple[int,int]]: List of tuples of (detected_changepoint, 
            actual_changepoint) assignments
    """

    def buildProb_NoObjective(sense):
        """
            Builds the optimization problem, minus the Objective function. 
            Makes multi-objective optimization simple
        """
        prob = LpProblem("Changepoint_Assignment", sense)

        # Create a variable for each pair of detected and actual changepoints
        # Assign detected changepoint dp to actual changepoint ap?
        vars = LpVariable.dicts(
            "x", (detected_changepoints, actual_changepoints), 0, 1, LpBinary)

        # Flatten vars into dict of tuples of keys
        x = {
            (dc, ap): vars[dc][ap] for dc in detected_changepoints for ap in
            actual_changepoints
        }

        ####### Constraints #########
        unique_count = 1
        # Only assign at most one changepoint to each actual changepoint
        for ap in actual_changepoints:
            prob += (
                lpSum(x[dp, ap] for dp in detected_changepoints) <= 1,
                f"Only_One_Changepoint_Per_Actual_Changepoint : {ap}"
            )
        # Each detected changepoint is assigned to at most one actual changepoint
        for dp in detected_changepoints:
            prob += (
                lpSum(x[dp, ap] for ap in actual_changepoints) <= 1,
                f"Only_One_Actual_Changepoint_Per_Detected_Changepoint : {dp}"
            )
        # Distance between chosen tuples must be within lag window
        for dp in detected_changepoints:
            for ap in actual_changepoints:
                # contraint for start point
                prob += (x[dp, ap] * abs(dp[0] - ap[0]) <= lag_window,
                            f"Distance_Within_Lag_Window : \
                            {dp[0]}_{ap[0]}_{unique_count}"
                            )
                unique_count += 1
                # constraint for end point
                prob += (x[dp, ap] * abs(dp[1] - ap[1]) <= lag_window,
                            f"Distance_Within_Lag_Window : \
                            {dp[1]}_{ap[1]}_{unique_count}"
                            )
                unique_count += 1
        return prob, x

    solver = PULP_CBC_CMD(msg=0)

    # Multi-Objective Optimization: First maximize number of assignments to find out
    # the best True Positive number that can be achieved
    # Find the largest number of change points:
    prob1, prob1_vars = buildProb_NoObjective(LpMaximize)
    prob1 += (
        lpSum(
            # Minimize the squared distance between assigned changepoints
            prob1_vars[dp, ap]
            for dp in detected_changepoints for ap in actual_changepoints
        ),
        "Maximize number of assignments"
    )
    prob1.solve(solver)
    # Calculate number of TP
    num_tp = len([
        (dp, ap)
        for dp in detected_changepoints for ap in actual_changepoints
        if prob1_vars[dp, ap].varValue == 1
    ])

    # Multi-Objective Optimization: Now minimize the squared distance between assigned
    # changepoints, using this maximal number of assignments
    # Use this number as the number of assignments for second optimization
    prob2, prob2_vars = buildProb_NoObjective(LpMinimize)
    prob2 += (
        lpSum(
            # Minimize the squared distance between assigned start points
            prob2_vars[dp, ap] * pow(dp[0] - ap[0], 2)
            for dp in detected_changepoints for ap in actual_changepoints
        ),
        "Squared_Distances_Start"
    )

    prob2 += (
        lpSum(
            # Minimize the squared distance between assigned end points
            prob2_vars[dp, ap] * pow(dp[1] - ap[1], 2)
            for dp in detected_changepoints for ap in actual_changepoints
        ),
        "Squared_Distances_End"
    )

    # Number of assignments is the number of true positives we found in the first
    # optimization
    prob2 += (
        lpSum(
            prob2_vars[dp, ap]
            for dp in detected_changepoints for ap in actual_changepoints
        ) == num_tp,
        "Maximize Number of Assignments"
    )
    prob2.solve(solver)
    return [
        (dp, ap)
        for dp in detected_changepoints for ap in actual_changepoints
        if prob2_vars[dp, ap].varValue == 1
    ]