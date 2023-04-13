"""
main.py
Nathan Brito, Shreya Thalvayapati, Yidi Wang, Lijun Zhang
Course: DS 3500 / Project Name: An Evolutionary Approach to TA/Lab Assignments
Homework Number 4
Date Created Mar 13, 2023 / Date Last Updated Mar 28, 2023
"""
from evo import Evo
from objectives import overallocation, conflicts, undersupport, unwilling, unpreferred
from agents import swapper, random_ta_reassignment, fix_overallocation, fix_unwilling, fix_unpreferred
from initial_solution import generate_initial_sol
import numpy as np
import pandas as pd

tas = np.loadtxt(open("tas.csv", "rb"), dtype="str", delimiter=",", skiprows=1)
sections = pd.read_csv('sections.csv')


def main():

    # create an Evo framework
    allocating_tas = Evo()

    # register some objectives
    allocating_tas.add_fitness_criteria("overallocation", overallocation)
    allocating_tas.add_fitness_criteria("conflicts", conflicts)
    allocating_tas.add_fitness_criteria("under supported", undersupport)
    allocating_tas.add_fitness_criteria("unwilling", unwilling)
    allocating_tas.add_fitness_criteria("not preferred", unpreferred)

    # register some agents
    allocating_tas.add_agent("swapper", swapper)
    allocating_tas.add_agent("random reassignment", random_ta_reassignment)
    allocating_tas.add_agent("fixing overallocation", fix_overallocation)
    allocating_tas.add_agent("fixing unwilling", fix_unwilling)
    allocating_tas.add_agent("fixing unpreferred", fix_unpreferred)

    # create a random initial solution
    initial_solution = generate_initial_sol()

    # seed the population with this solution
    allocating_tas.add_solution(initial_solution)
    print(allocating_tas)

    # evolve a 100,000 times
    # remove dominated solutions after every 100 iterations
    # outputting the population every 1,000 iterations
    allocating_tas.evolve(100000, 100, 1000)

    # export results into a csv
    allocating_tas.export_results()

    # print the final results
    print(allocating_tas)


if __name__ == '__main__':
    main()
