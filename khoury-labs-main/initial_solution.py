"""
initial_solution.py
Nathan Brito, Shreya Thalvayapati, Yidi Wang, Lijun Zhang
Course: DS 3500 / Project Name: An Evolutionary Approach to TA/Lab Assignments
Homework Number 4
Date Created Mar 27, 2023 / Date Last Updated Mar 27, 2023
"""
import numpy as np

# import tas information as a global variable
tas = np.loadtxt(open("tas.csv", "rb"), dtype="str", delimiter=",", skiprows=1)


def generate_initial_sol():
    """ generates the initial possible solution by assigning TAs to their preferred sections
    
    Returns: 
        - ta_assignments (2D numpy array): just assigns 1s to preferred 
                                           sections and 0s everywhere else
    """
    # slice first three columns of TAs to get TAs preferences 
    ta_preferences = tas[:, 3:]
    
    # create a zero 2d numpy array the size of ta_preferences
    ta_assignments = np.zeros(ta_preferences.shape, dtype="int")
    
    # assign TAs to their preferred sections
    ta_assignments[(ta_preferences == "P")] = 1
    
    # return TA assignments
    return ta_assignments

