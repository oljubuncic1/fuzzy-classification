import permutation_ga as pga
from myunittest import Test



class GA_Test(Test):
    
    def test_not_run_exception(self):
        try:
            pga_instance = pga.PermuatationGA()
            pga_instance.get_best()
        except Exception as ex:
            self.should_eq(
                "best should not be accessed without previous runs", 
                type(ex).__name__, 
                "AssertionError"
            )

    def test_init_pop(self):
        pga_instance = pga.PermuatationGA()
        values = range(10)
        init_pop_size = 2
        sample_size = 5
        init_pop = pga_instance.init_pop(values, init_pop_size, sample_size)

        self.should_eq(
            "should return appropriate sized initial pop value 10",
            len(init_pop),
            init_pop_size
        )

        try:
            init_pop_size = 0
            init_pop = pga_instance.init_pop([1, 2, 3], init_pop_size, sample_size)
        except Exception as ex:
            self.should_eq(
                "should not accept zero",
                type(ex).__name__,
                "ValueError"
            )

        try:
            init_pop_size = -10
            init_pop = pga_instance.init_pop([1, 2, 3], init_pop_size, sample_size)
        except Exception as ex:
            self.should_eq(
                "should not accept negative values",
                type(ex).__name__,
                "ValueError"
            )

    def test_crossover(self):
        pga_instance = pga.PermuatationGA()
        values = range(10)
        init_pop_size = 5
        sample_size = 2
        crossover_perc = 0.5
        objective_function = lambda x: len(x)

        pop = pga_instance.init_pop(values, init_pop_size, sample_size)
        pop = pga_instance.crossover(pop, crossover_perc, objective_function)

        self.should_eq(
            "population should grow by crossover perc", 
            len(pop), 
            int( (1 + crossover_perc) * init_pop_size)
        )

    def test_crossover_single(self):
        return

    def test_select(self):
        return


def test():
    test_suite = GA_Test()
    test_suite.main()

if __name__ == "__main__":
    test()
