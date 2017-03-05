import random
from dummy_logger import DummyLogger
import logging



class FeatureSelectionGA():
    def __init__(self, feature_n, objective_function, init_pop_n=4, generation_n=10, mutation_probability=0.05, selection_pressure=1.5, crossover_perc=0.5):
        self.is_run = False
        self.best = None 
        self.feature_n = feature_n
        self.mutation_probability = mutation_probability
        self.logger = DummyLogger()

        self.objective_function = objective_function
        self.init_pop_n = init_pop_n
        self.generation_n = generation_n
        self.selection_pressure = selection_pressure
        self.crossover_perc = crossover_perc
    
    def set_logger(self, logger):
        self.logger = logger

    def __generate_random_individual(self):
        individual = []
        for i in range(self.feature_n):
            if random.random() < 0.5:
                individual.append(1)
            else:
                individual.append(0)

        return individual
    
    def features_objective_function(self, x):
        return self.objective_function( self.binary_to_features(x) )

    def init_pop(self):
        self.pop = []
        for i in range(self.init_pop_n):
            self.pop.append(self.__generate_random_individual()) 

    def rank_select(self):
        v = int( random.random() * ( len(self.pop) ** ( 1 / self.selection_pressure) ) )
        pop = sorted(self.pop, key=lambda x: self.features_objective_function(x), reverse=True)

        return pop[v]

    def select(self):
        selection = []
        for i in range( int(self.crossover_perc * len(self.pop)) ):
            first_parent = self.rank_select()
            second_parent = self.rank_select()
            
            selection.append( (first_parent, second_parent) )
        
        return selection

    def crossover_single(self, first_parent, second_parent):
        child = [ first_parent[i] ^ second_parent[i] for i in range(len(first_parent)) ]
        return child

    def crossover(self):
        selection = self.select()
        for i in range(len(selection)):
            first_parent = selection[i][0]
            second_parent = selection[i][1]
            child = self.crossover_single(first_parent, second_parent)
            self.pop.append(child)

    def mutate_single(self, p):
        for i in range(len(p)):
            if random.random() < self.mutation_probability:
                p[i] = 1 - p[i]

        return p

    def mutate(self):
        for i in range(len(self.pop)):
            if i > 0:
                self.pop[i] = self.mutate_single(self.pop[i])

    def binary_to_features(self, binary_string):
        features = []
        for i in range( len(binary_string) ):
            if binary_string[i] == 1:
                features.append(i)

        return features

    def run(self):
        self.init_pop()

        for i in range(self.generation_n):
            self.logger.info("GA generation " + str(i))

            self.logger.debug("Crossing over...")
            self.crossover()

            self.pop = sorted(self.pop, key=self.features_objective_function, reverse=True)
            
            self.mutate()
            self.pop = self.pop[:self.init_pop_n]

        self.best = self.pop[0]
        self.is_run = True
        
    def get_best(self):
        if not self.is_run:
            raise AssertionError("Algorithm not run on any values")
        else:
            return self.binary_to_features(self.best)
