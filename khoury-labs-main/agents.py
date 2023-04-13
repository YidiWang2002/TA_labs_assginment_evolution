"""
agents.py
Nathan Brito, Shreya Thalvayapati, Yidi Wang, Lijun Zhang
Course: DS 3500 / Project Name: An Evolutionary Approach to TA/Lab Assignments
Homework Number 4
Date Created Mar 24, 2023 / Date Last Updated Mar 28, 2023
"""
import random
import numpy as np
import pandas as pd

# import tas and sections information as a global variable
tas = np.loadtxt(open("tas.csv", "rb"), dtype="str", delimiter=",", skiprows=1)
sections = pd.read_csv('sections.csv')


def generate_indices(num_rows, num_cols):
    """ generates random indices to produce a 3 x 3 sub-array

    Arguments:
      - num_rows (int): the number of rows the original 2D numpy array has
      - num_cols (int): the number of columns the original 2D numpy array has

    Returns:
      - starting row (int): index for the first row to slice
      - ending row (int): index for the last row to slice
      - starting column (int): index for the first column to slice
      - ending column (int): index for the last column to slice
    """

    assert num_rows >= 3, "original array needs to have at least 3 rows"
    assert num_cols >= 3, "original array needs to have at least 3 columns"

    # generate random indices for the rows to extract
    starting_row = np.random.randint(0, (num_rows - 3))
    ending_row = starting_row + 3

    # generate random indices for the columns to extract
    starting_col = np.random.randint(0, (num_cols - 3))
    ending_col = starting_col + 3

    # returning four ints, each an index
    return starting_row, ending_row, starting_col, ending_col


def swapper(solutions_list):
    """ swaps the positions of two 3 x 3 sub-arrays in the original 2D numpy array

    Arguments:
      - solutions_list (3D numpy array): a list of 2D numpy arrays, each a solution

    Returns:
      - solution (2D numpy array): modified version after 2 sub-arrays have been swapped
    """

    # grab just one solution from the list
    solution = solutions_list[-1]

    # extract the number of rows and columns
    num_rows, num_cols = solution.shape

    # generate the first random subset
    first_start_row, first_end_row, first_start_col, first_end_col = generate_indices(num_rows, num_cols)
    first_subset = solution[first_start_row: first_end_row,
                            first_start_col: first_end_col]

    # generate the second random subset
    second_start_row, second_end_row, second_start_col, second_end_col = generate_indices(num_rows, num_cols)
    second_subset = solution[second_start_row: second_end_row,
                             second_start_col: second_end_col]

    # make deep copies of the original subsets
    first_subset_copy = np.copy(first_subset)
    second_subset_copy = np.copy(second_subset)

    # swap the positions of the subsets in the original array
    solution[first_start_row: first_end_row, first_start_col: first_end_col] = second_subset_copy
    solution[second_start_row: second_end_row, second_start_col: second_end_col] = first_subset_copy

    return solution


def random_ta_reassignment(solutions_list):
    """ randomly adds or removes a TA for each section in the given test array

    Arguments:
        solutions_list (3D numpy array): a list of 2D numpy arrays, each a solution

    Returns: the updated test array after randomly adding or removing TAs for each section
    """

    # grab just one solution from the list
    solution = solutions_list[-1]

    # variable for number of sections
    num_sections = solution.shape[1]

    # iterate through each section in the test array
    for section in range(num_sections):

        # randomly decide whether to add or remove a TA from the current section
        action = np.random.choice(['add', 'remove'])

        # if we are adding a TA
        if action == 'add':
            # randomly choose a TA to add to the section and set their assignment value to 1
            ta_to_add = np.random.randint(0, solution.shape[0])
            solution[ta_to_add][section] = 1
        # if we are removing a TA
        else:
            # find all TAs currently assigned to the section
            assigned_tas = np.where(solution[:, section] == 1)[0]
            # if there are any assigned TAs, randomly choose one to remove and set their assignment value to 0
            if len(assigned_tas) > 0:
                ta_to_remove = np.random.choice(assigned_tas)
                solution[ta_to_remove][section] = 0

    return solution


def fix_overallocation(solutions_list):
    """ removes a section from a TA's row of preference if they reached their max

    Arguments:
        - solutions_list (3D numpy array): a list of 2D numpy arrays, each a solution

    Returns: updated assignment
    """

    # grab just one solution
    solution = solutions_list[-1]

    # get the max sections from TA file
    max_sections = tas.T[2].astype(np.int)

    # loop through each row and max_section
    for row, max_section in zip(solution, max_sections):
        # if sections assign for a TA is greater than their max sections
        if sum(row) > max_section:
            # select a random index of a 1 value in the list
            idx = random.choice([i for i, v in enumerate(row) if v == 1])
            # switch the selected value from 1 to 0
            row[idx] = 0

    # return TA assignment
    return solution


def fix_unwilling(solutions_list):
    """
    Updates the test array by setting the value to 0 for the sections where the TA is marked as Unavailable.

    Args:
        - solutions_list (3D numpy array): a list of 2D numpy arrays, each a solution

    Returns: an updated solutions array after setting the value to 0 for sections where the TA is Unavailable
    """

    # grab just one solution
    solution = solutions_list[-1]

    # extract TA's preferences from the tas array
    tas_prefs = tas[:, 3:]

    # iterate through all the rows in the test array and set the value to 0 for sections
    # where the TA is marked as Unavailable
    solution[np.logical_and(solution == 1, tas_prefs == 'U')] = 0

    return solution


def fix_unpreferred(solutions_list):
    """ updates the test array by setting the value to 0 for the sections the TA does not prefer

    Arguments:
        - solutions_list (3D numpy array): a list of 2D numpy arrays, each a solution

    Returns: the updated test array after setting the value to 0 for sections the TA does not prefer
    """
    
    # grab just one solution
    solution = solutions_list[-1]

    # extract TA's preferences from the tas array
    ta_prefs = tas[:, 3:]

    # iterate through all the rows in the test array and set the value to 0 for sections
    # where the TA is marked as Willing
    solution[np.logical_and(solution == 1, ta_prefs == 'W')] = 0
    
    return solution
