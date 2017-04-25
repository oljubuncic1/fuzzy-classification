
#ifndef CPP_TEST_RANDOMFOREST_H
#define CPP_TEST_RANDOMFOREST_H

#include "includes.h"
#include "RandomFuzzyTree.h"

#include <cassert>

void test_all_numeric() {
    function<vector<int>(void)> rfg = []() -> vector<int> {
        static int i = 0;

        static vector<vector<int>> feature_sets = {
                {0, 1},
                {2, 3}
        };

        return feature_sets[i++];
    };

    RandomFuzzyTree rft;
    rft.set_random_feature_generator(rfg);

    data_t data = {
            { {1, 2, 3}, "2" }
    };
    vector<range_t> ranges = {
            {1, 5}
    };

    rft.fit(data, ranges);
}

void run_all_tests() {
    test_all_numeric();
}

#endif //CPP_TEST_RANDOMFOREST_H
