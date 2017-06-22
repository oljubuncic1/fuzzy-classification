
#ifndef CPP_TEST_RANDOMFOREST_H
#define CPP_TEST_RANDOMFOREST_H

void testFastRandomFuzzyTree();

#include "includes.h"
#include "RandomFuzzyTree.h"
#include "FastRandomFuzzyTree.h"

#include <cassert>

void run_all_tests() {
    testFastRandomFuzzyTree();

    cout << "All tests passed" << endl;
}

void testFastRandomFuzzyTree() {
    vector<int> categorical = {1, 2, 3};
    vector<int> numerical = {4, 5, 6};

    data_t data = {
            {{0.1, 0.1, 3, 4}, "1"},
            {{0.2, 0.6, 3, 4}, "1"},
            {{0.3, 1.2, 3, 4}, "1"},
            {{0.6, 0.2, 3, 4}, "2"},
            {{0.7, 0.7, 3, 4}, "2"},
            {{0.8, 1.3, 3, 4}, "2"},
            {{1.2, 0.3, 3, 4}, "3"},
            {{1.3, 0.8, 3, 4}, "3"},
            {{1.4, 1.4, 3, 4}, "3"},
    };
    vector<range_t > ranges = {
            {0, 1.4},
            {0, 1.4},
    };
    vector<int> categorical_features = {};

    FastRandomFuzzyTree frft;
    frft.set_verbose(2);
    frft.fit(data, ranges, categorical);
}

#endif //CPP_TEST_RANDOMFOREST_H
