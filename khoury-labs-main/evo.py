"""
evo.py
Nathan Brito, Shreya Thalvayapati, Yidi Wang, Lijun Zhang
Course: DS 3500 / Project Name: An Evolutionary Approach to TA/Lab Assignments
Homework Number 4
Date Created Mar 13, 2023 / Date Last Updated Mar 23, 2023
"""
# Source for timer: https://www.udacity.com/blog/2021/09/create-a-timer-in-python-step-by-step-guide.html
import random as rnd
import copy
from functools import reduce
import time
from create_summary_table import export_summary

class Evo:

    def __init__(self):

        # - a dictionary for the population
        # - the keys of the dictionary is going to be a tuple of tuples: ((obj1, eval1), (obj2, eval2), (obj3, eval3))
        # - remember that keys need to be immutable objects!
        # - the keys here can be tuples (because there are multiple objective functions that all output scores)
        # - the values of the dictionary are the actual solutions themselves
        # - treating solutions that have the same scores as duplicates because if the underlying
        #   differences were important, you ought to have an additional objective that characterizes that
        self.pop = {}
        # - a dictionary for storing the objective functions
        # - the keys are the names of the objective functions
        # - the values are the actual objective functions themselves
        self.fitness = {}
        # - agents are the functions that tweak our solutions to make them stronger
        # - the keys of the dictionary are the names of the agents
        # - the values of the dictionary are the actual agent functions
        # - we are also going to be storing the # of input functions that each agent takes in
        # - while for this assignment, each agent will be outputting just one solution,
        #   it may take in more than 1 solution as input and merge them in some way
        # - format of the values are tuples: (agent operator, # input solutions)
        self.agents = {}

    def size(self):
        """ returns the length of the population dictionary """
        return len(self.pop)

    def add_fitness_criteria(self, name, f):
        """ adds objective functions to the internal fitness dictionary of the framework

        Arguments:
            - name (string): what the objective function is called
            - f (function): an actual objective function

        Returns: nothing, just updates the internal state of the
                 framework with new objective functions
        """
        self.fitness[name] = f

    def add_agent(self, name, op, k=1):
        """ adds agent functions to the internal agents dictionary of the framework

        Arguments:
            - name (string): what the agent function is called
            - op (function): the agent function itself
            - k (int): the number of solutions the agent function takes in,
                       the default is just 1

        Returns: nothing, just updates the internal state of
                 the framework with new agent functions
        """
        self.agents[name] = (op, k)

    def get_random_solutions(self, k):
        """ pick k random solutions from the population as a list of solutions

        Arguments:
            - k (int): number of random solutions we are picking

        Returns: a list of randomly-picked solutions
        """
        # if for some reason there are no solutions in the population
        # (this really shouldn't be happening though)
        if self.size == 0:
            return []
        else:
            # just get the solutions themselves (not the tuple of scores) from the population dictionary
            pop_vals = tuple(self.pop.values())
            # this is some dense code on line 82, so here's a debrief:
            # - we use deepcopy because it allows us to change a copy of an object without changing the original
            # - in the case of evolutionary computing, this means that we
            #   can change the 'child' without changing the 'parent'
            # - we are picking a random solution for the list of solutions (pop_vals) k number of times
            # - each time we pick a random solution, we are making a deep copy of it
            return [copy.deepcopy(rnd.choice(pop_vals)) for _ in range(k)]

    def add_solution(self, sol):
        """ add a new solution to the population

        Arguments:
            - sol (2D numpy array): a single solution

        Returns: nothing, just adds to the population dictionary
        """
        # this is some dense code on line 95, so let's break it down a bit:
        #   - we know that the fitness dictionary has names of the objective functions
        #     as the keys and the actual functions themselves as the values
        #   - we are going to be running every single objective function against the solution
        #   - as each objective function runs, we are creating a tuple with the name of the objective function
        #     in the 0th position and the score that objective function outputs in the 1st position
        #   - then, we are making whole thing a tuple so that it is tuple of tuples
        #     rather than a list of tuples (which are mutable, and therefore not allowed to be keys of dictionaries)
        evaluation = tuple([(name, f(sol)) for name, f in self.fitness.items()])
        # then we are adding that solution to the population
        # by design, the dictionary will only be updated if the evaluation score was different
        self.pop[evaluation] = sol

    def run_agent(self, name):
        """ invoke an agent function against the current population

        Arguments:
            - name (string): the name of the agent function to run

        Returns: nothing, just runs the chosen agent function on a set of random solutions
        """
        # fetch the actual agent function and the number of inputs it takes in
        op, k = self.agents[name]
        # fetch some random solutions
        picks = self.get_random_solutions(k)
        # run the agent on the random solutions to potentially make them even better
        new_solution = op(picks)
        # add the new population to the framework
        self.add_solution(new_solution)

    def evolve(self, n=1, dom=100, status=100, time_limit=600):
        """ runs n random agent functions against the population

        Arguments:
            - n (int): the number of agent invocations
            - dom (int): number of iterations between discarding the dominated solutions,
                         the reason we don't get rid of dominated solutions immediately is because
                         bad solutions may actually lead to better ones later on,
                         we need to allow some variation to develop before getting rid of the weaklings

        Returns: nothing, just ties the pieces of the framework together
        """

        # list of the agent functions' names
        agent_names = list(self.agents.keys())

        # start the timer
        start_time = time.time()

        for i in range(n):
            # if the time limit hasn't been reached yet
            if time.time() - start_time < time_limit:
                # pick a random agent function to run
                pick = rnd.choice(agent_names)
                # run the agent function
                self.run_agent(pick)
                # if it has been a while since we got rid of the dominated values, remove them
                if i % dom == 0:
                    self.remove_dominated()
                # occasionally view details of the population
                if i % status == 0:
                    # only want to see the nds
                    self.remove_dominated()
                    print("Iteration: ", i)
                    print("Population Size: ", self.size())
                    # utilizes the __str__ method
                    print(self)
            else:
                break

        # raise error if it's past the time limit
        if time.time() - start_time >= time_limit:
            raise TimeoutError

        # clean up the population one last time
        self.remove_dominated()

    @staticmethod
    def _dominates(first_solution_set, second_solution_set):
        """ determines whether one solution dominates another

        Arguments:
            - first_solution_set (tuple of tuples):
            - second_solution_set (tuple of tuples):

        Returns: a boolean, whether the first_solution_set dominates the second_solution_set
        """
        # extract out the scores from the two passed in population dictionaries
        first_set_scores = [score for _, score in first_solution_set]
        second_set_scores = [score for _, score in second_solution_set]

        # creating a list of value differences between the two sets of scores
        score_diffs = list(map(lambda x, y: y - x, first_set_scores, second_set_scores))

        # find the min and max score differences
        min_diff = min(score_diffs)
        max_diff = max(score_diffs)

        # a solution needs to meet two criteria to be considered "dominating":
        #   1) it needs to be at least as good with respect to every criteria
        #   2) it needs to be STRICTLY better than at least one other solution
        return min_diff >= 0.0 and max_diff > 0

    @staticmethod
    def _reduce_nds(S, p):
        """ a reducer function that removes dominated from a set

        Arguments:
            - S: a set of solutions
            - p: one particular solution

        Returns: a set of non-dominated-solutions
        """
        return S - {q for q in S if Evo._dominates(p, q)}

    def remove_dominated(self):
        """ removes dominated solutions from a solution set """
        nds = reduce(Evo._reduce_nds, self.pop.keys(), self.pop.keys())
        # makes sure that the population only contains nds
        self.pop = {k: self.pop[k] for k in nds}
    
    def __str__(self):
        """ outputs the solutions in the population

        Returns: a string representation of the population dictionary
        """
        # an empty string
        result = ""

        # add to the string
        for evaluation, solutions in self.pop.items():
            result += str(dict(evaluation)) + ": " + str(solutions) + "\n"

        # display the string
        return result

    def export_results(self):
        """ export non-dominated results in a csv format """
        export_summary(list(self.pop.keys()))
