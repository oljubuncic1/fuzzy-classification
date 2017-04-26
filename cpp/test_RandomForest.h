
#ifndef CPP_TEST_RANDOMFOREST_H
#define CPP_TEST_RANDOMFOREST_H

#include "includes.h"
#include "RandomFuzzyTree.h"

#include <cassert>

void test_all_numeric() {
    function<vector<int>(void)> rfg = []() -> vector<int> {
        static int i = 0;

        vector<vector<int>> feature_sets = {
                {0, 1},
        };

        return feature_sets[i++];
    };

    RandomFuzzyTree rft;
    rft.set_random_feature_generator(rfg);
    rft.set_max_depth(0);

    data_t data = {
            { {0.1, 0.1, 3, 4}, "1" },
            { {0.2, 0.6, 3, 4}, "1" },
            { {0.3, 1.2, 3, 4}, "1" },
            { {0.6, 0.2, 3, 4}, "2" },
            { {0.7, 0.7, 3, 4}, "2" },
            { {0.8, 1.3, 3, 4}, "2" },
            { {1.2, 0.3, 3, 4}, "3" },
            { {1.3, 0.8, 3, 4}, "3" },
            { {1.4, 1.4, 3, 4}, "3" },
    };
    vector<range_t> ranges = {
            {0, 1.4},
            {0, 1.4},
    };

    vector<int> categorical_features = {};
    vector<int> numerical_features = {0, 1};

    rft.fit(data,
            ranges,
            categorical_features,
            numerical_features);

    return;
}

void run_all_tests() {
    test_all_numeric();

    cout << "All tests passed" << endl;
}

#endif //CPP_TEST_RANDOMFOREST_H
