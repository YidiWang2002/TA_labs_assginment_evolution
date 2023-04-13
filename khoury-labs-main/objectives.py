"""
objectives.py
Nathan Brito, Shreya Thalvayapati, Yidi Wang, Lijun Zhang
Course: DS 3500 / Project Name: An Evolutionary Approach to TA/Lab Assignments
Homework Number 4
Date Created Mar 20, 2023 / Date Last Updated Mar 28, 2023
"""

import numpy as np
import pandas as pd

# Import ta and section information as global variables
tas = np.loadtxt(open("tas.csv", "rb"), dtype="str", delimiter=",", skiprows=1)
sections = pd.read_csv('sections.csv')


def overallocation(assignments):
    """ calculates the number of times a TA has been assigned to more than their maximum sections
    
    Arguments: 
        - assignments (2D np array): Information about TAs and their assigned sections

    Returns: the overallocation penalty
    """
    # Get the max sections from TA file
    max_sections = tas.T[2].astype(np.int)

    # apply the lambda function to each row of the array and the corresponding element from the list
    diffs = map(lambda row, num: np.sum(row) - num if np.sum(row) > num else 0, assignments, max_sections)
    
    # Return overallocation penalty 
    return sum(diffs)


def conflicts(assignments):
    """ counts number of TAs that be assigned to conflicted section
    
    Arguments:
        - assignments: (2d np array) Result that shows TAs and their assigned sections
    
    Returns: conflicts (int), the number of conflicts
    """
    
    # initialize the number of conflicts
    num_conflicts = 0
    
    # test stored as a 2D numpy array
    for ta in range(assignments.shape[0]):
        # extract the TA assigned time, and sections is a df
        time = sections.daytime[assignments[ta] == 1]
        # if there are duplicated time sections
        if len(list(time)) > len(set(time)):
            # counted as one more conflicts
            num_conflicts += 1

    return num_conflicts


def undersupport(assignments):
    """ calculates the total penalty score for undersupport of sections
    
    Arguments:
        - assignments: (2d np array) Result that shows TAs to their assigned sections
    
    Returns: the total penalty score for undersupport of sections
    """
    # calculate total assigned TAs for each section
    total_assigned = assignments.sum(axis=0)
    
    # calculate penalty scores
    scores = np.maximum(sections['min_ta'].values - total_assigned, 0)
    
    # return the total scores of undersupport
    return np.sum(scores)


def unwilling(assignments):
    """ calculates the number of times a TA has been assigned to a section but is unwilling to work
    
    Arguments: 
        - assignments (2D np array): Information about TAs and their assigned sections

    Returns: the unwilling penalty
    """
    # slice the first three columns of the TA data to get only TAs' preferences
    ta_preferences = tas[:, 3:]

    # create a Boolean array that indicates where the TA is assigned to a section but is unwilling
    mask = np.logical_and(assignments == 1, ta_preferences == "U")

    # return count the number of True values in the resulting bool array (unwilling penalty)
    return np.count_nonzero(mask)


def unpreferred(assignments):
    """ calculates the number of times a TA has been assigned a section but is not preferred

    Arguments:
        - assignments (2D np array): Information about TAs and their assigned sections

    Returns: the unpreferred penalty
    """
    # slice the first three columns of the TA data to get only TAs' preferences
    ta_preferences = tas[:, 3:]

    # create a Boolean array that indicates where the TA is willing and assigned to a section
    mask = np.logical_and(assignments == 1, ta_preferences == "W")

    # return count the number of True values in the resulting bool array (unpreferred penalty)
    return np.count_nonzero(mask)
