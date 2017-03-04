import random
from dummy_logger import DummyLogger



class PermuatationGA():
    
    def __init__(self, values):
        self.is_run = False
        self.best = None 
        self.default_init_pop_size = 10
        self.values = values
        self.mutation_probability = 0.05
        self.logger = DummyLogger()
    
    def set_logger(self, logger):
        self.logger = copy(logger)

    def init_pop(self, pop_size, sample_size):
        if pop_size <= 0 or sample_size <= 0:
            raise ValueError(
                "You should send a positive init pop size and sample size"
            )
        
        pop = []
        for i in range(pop_size):
            pop.append(random.sample(self.values, sample_size))
        
        return pop

    def rank_select(self, pop, objective_function, selection_factor = 1.5):
        v = int( random.random() * ( len(pop) ** ( 1 / selection_factor) ) )
        pop = sorted(pop, key=lambda x: objective_function(x), reverse=True)

        return pop[v]

    def select(self, pop, crossover_perc, objective_function):
        selection = []
        for i in range( int(crossover_perc * len(pop)) ):
            first_parent = self.rank_select(pop, objective_function)
            second_parent = self.rank_select(pop, objective_function)
            
            selection.append( (first_parent, second_parent) )
        
        return selection

    def crossover_single(self, first_parent, second_parent, objective_function, mplets_cnt=3):
        children = []

        for i in range(mplets_cnt):
            child = list( set(first_parent).intersection(second_parent) )
            left = list( set(first_parent).union(set(second_parent)) - set(child) )
            random.shuffle(left) 
            i = 0
            while len(child) < len(first_parent):
                child.append(left[i])
                i += 1
            children.append(child)

        best_child = max(children, key=objective_function)

        return best_child

    def crossover(self, pop, crossover_percent, objective_function):
        selection = self.select(pop, crossover_percent, objective_function)
        for i in range(len(selection)):
            first_parent = selection[i][0]
            second_parent = selection[i][1]
            pop.append(self.crossover_single(
                first_parent, second_parent, objective_function)
            )
        
        return pop

    def mutate_single(self, p):
        if random.random() < self.mutation_probability:
            not_in_p = list( set(self.values) - set(p) )
            
            random.shuffle(p)
            random.shuffle(not_in_p)

            p[0] = not_in_p[0]

        return p

    def mutate(self, pop):
        for i in range(len(pop)):
            pop[i] = self.mutate_single(pop[i])
        
        return pop

    def run(
        self,
        objective_function, 
        init_pop_size = 4, 
        sample_size=2,
        crossover_percent=0.5,
        generation_cnt = 10
    ):

        pop = self.init_pop(init_pop_size, sample_size)

        for i in range(generation_cnt):
            self.logger.info("GA generation " + str(i))

            self.logger.debug("Crossing over...")
            pop = self.crossover(pop, crossover_percent, objective_function)

            pop = sorted(pop, key=lambda x: objective_function(x), reverse=True)
            
            pop = self.mutate(pop[1:])
            pop = pop[:init_pop_size]
            

        self.best = pop[0]
        self.is_run = True
        
    def get_best(self):
        if not self.is_run:
            raise AssertionError("Algorithm not run on any values")
        else:
            return self.best
