""" create_summary_table.py
Nathan Brito, Shreya Thalvayapati, Yidi Wang, Lijun Zhang
Course: DS 3500 / Project Name: An Evolutionary Approach to TA/Lab Assignments
Homework Number 4
Date Created Mar 28, 2023 / Date Last Updated Mar 28, 2023
"""
import pandas as pd


def export_summary(population):
    """ creates csv file for non-dominated solutions
    
    Arguments: 
        - population: list of keys of non-dominated population
        
    Returns: csv file with non-dominated soltuions
    """

    # convert tuples to dicts from population
    list_of_dicts = [dict(t) for t in population]

    # get results from population dict into a pd df
    df = pd.DataFrame(list(list_of_dicts))

    # rename some columns to match assignment's description
    df = df.rename(columns={"under supported": "undersupport", "not preferred" :"unpreferred"})

    # assign group name column in df
    df = df.assign(groupname="MagicEVO")

    # reorder columns to have groupname first
    df = df[["groupname", "overallocation", "conflicts", "undersupport", "unwilling", "unpreferred"]]

    # export df into csv
    df.to_csv("summary_table.csv", index=False)
