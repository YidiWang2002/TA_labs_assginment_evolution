"""
tests.py
Nathan Brito, Shreya Thalvayapati, Yidi Wang, Lijun Zhang
Course: DS 3500 / Project Name: An Evolutionary Approach to TA/Lab Assignments
Homework Number 4
Date Created Mar 13, 2023 / Date Last Updated Mar 27, 2023
"""
import pytest
import numpy as np
import pandas as pd
from objectives import overallocation, conflicts, undersupport, unwilling, unpreferred


class TestClass:
    def test_one(self):
        """ Test objective functions with test 1 """
        # Import test 1
        test = np.loadtxt(open("test1.csv", "rb"), dtype="int", delimiter=",")
        
        # Test objective functions
        assert overallocation(assignments=test) == 37, "Overallocation is incorrect."
        assert conflicts(assignments=test) == 8, "Conflicts is incorrect."
        assert undersupport(assignments=test) == 1, "Undersupport is incorrect."
        assert unwilling(assignments=test) == 53, "Unwilling is incorrect."
        assert unpreferred(assignments=test) == 15, "Unpreferred is incorrect."

    def test_two(self):
        """ Test objective functions with test 2 """
        # Import test 2
        test = np.loadtxt(open("test2.csv", "rb"), dtype="int", delimiter=",")

        # Test objective functions
        assert overallocation(assignments=test) == 41, "Overallocation is incorrect."
        assert conflicts(assignments=test) == 5, "Conflicts is incorrect."
        assert undersupport(assignments=test) == 0, "Undersupport is incorrect."
        assert unwilling(assignments=test) == 58, "Unwilling is incorrect."
        assert unpreferred(assignments=test) == 19, "Unpreferred is incorrect."

    def test_three(self):
        """ Test objective functions with test 3 """
        # Import test 3
        test = np.loadtxt(open("test3.csv", "rb"), dtype="int", delimiter=",")
        
        # Test objective functions
        assert overallocation(assignments=test) == 23, "Overallocation is incorrect."
        assert conflicts(assignments=test) == 2, "Conflicts is incorrect."
        assert undersupport(assignments=test) == 7, "Undersupport is incorrect."
        assert unwilling(assignments=test) == 43, "Unwilling is incorrect."
        assert unpreferred(assignments=test) == 10, "Unpreferred is incorrect."
